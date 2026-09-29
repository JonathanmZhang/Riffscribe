"""MT3 (Magenta, multi-instrument checkpoint) on one bake-off clip, on CPU.
Runs in the bakeoff-mt3 image: python /bakeoff/run_mt3.py <segment id>

InferenceModel is the official Colab's wrapper
(mt3/colab/music_transcription_with_transformers.ipynb, Copyright 2021
Google LLC, Apache-2.0), trimmed to the mt3 model type. Notes are kept only
when their program is a GM guitar (24-31); the full output's recall is also
recorded.
"""

import functools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import gin  # noqa: E402
import jax  # noqa: E402
import note_seq  # noqa: E402
import numpy as np  # noqa: E402
import seqio  # noqa: E402
import t5  # noqa: E402
import t5x  # noqa: E402
import tensorflow.compat.v2 as tf  # noqa: E402
from mt3 import metrics_utils, models, network, note_sequences, preprocessors, spectrograms, vocabularies  # noqa: E402

from common import clip_path, write_notes  # noqa: E402

CHECKPOINT = "/opt/checkpoints/mt3/"
GIN_DIR = "/opt/mt3/mt3/gin"
SAMPLE_RATE = 16000
GUITAR_PROGRAMS = range(24, 32)


class InferenceModel:
    def __init__(self, checkpoint_path: str):
        self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        self.inputs_length = 256
        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=1))
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                [f"{GIN_DIR}/model.gin", f"{GIN_DIR}/mt3.gin"],
                ["from __gin__ import dynamic_registration", "from mt3 import vocabularies",
                 "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
                 "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS"],
                finalize_config=False)
        module = network.Transformer(config=gin.get_configurable(network.T5Config)())
        self.model = models.ContinuousInputsEncoderDecoderModel(
            module=module, input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config))
        initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def, init_fn=self.model.get_initial_variables,
            input_shapes={"encoder_input_tokens": (self.batch_size, self.inputs_length),
                          "decoder_input_tokens": (self.batch_size, self.outputs_length)},
            partitioner=self.partitioner)
        self._predict_fn = self._get_predict_fn(initializer.train_state_axes)
        self._train_state = initializer.from_checkpoint_or_scratch(
            [t5x.utils.RestoreCheckpointConfig(path=checkpoint_path, mode="specific", dtype="float32")],
            init_rng=jax.random.PRNGKey(0))

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(params, batch, decoder_params={"decode_rng": None})
        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec("data"), None),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"))

    def __call__(self, audio):
        frame_size = self.spectrogram_config.hop_width
        audio = np.pad(audio, [0, frame_size - len(audio) % frame_size], mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        times = np.arange(len(audio) // frame_size) / self.spectrogram_config.frames_per_second
        ds = tf.data.Dataset.from_tensors({"inputs": frames, "input_times": times})
        for pp in [
            functools.partial(t5.data.preprocessors.split_tokens_to_inputs_length,
                              sequence_length=self.sequence_length, output_features=self.output_features,
                              feature_key="inputs", additional_feature_keys=["input_times"]),
            preprocessors.add_dummy_targets,
            functools.partial(preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config),
        ]:
            ds = pp(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(ds, task_feature_lengths=self.sequence_length)
        model_ds = model_ds.batch(self.batch_size)
        inferences = (tokens for batch in model_ds.as_numpy_iterator()
                      for tokens in self.vocabulary.decode_tf(
                          self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(0))[0]).numpy())
        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            tokens = np.array(tokens, np.int32)
            if vocabularies.DECODED_EOS_ID in tokens:
                tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
            start = example["input_times"][0]
            start -= start % (1 / self.codec.steps_per_second)
            predictions.append({"est_tokens": tokens, "start_time": start, "raw_inputs": []})
        return metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec)["est_ns"]


def main(seg: str) -> None:
    t0 = time.perf_counter()
    model = InferenceModel(CHECKPOINT)
    load = time.perf_counter() - t0

    path = clip_path(seg)
    with open(path, "rb") as f:
        audio = note_seq.audio_io.wav_data_to_samples_librosa(f.read(), sample_rate=SAMPLE_RATE)
    model(audio)  # warm-up (includes JAX compilation), as for every model
    t0 = time.perf_counter()
    ns = model(audio)
    infer = time.perf_counter() - t0
    duration = len(audio) / SAMPLE_RATE

    as_dict = lambda ns_: [{"onset": n.start_time, "end": n.end_time, "midi": n.pitch} for n in ns_]  # noqa: E731
    pitched = [n for n in ns.notes if not n.is_drum]
    guitar = [n for n in pitched if n.program in GUITAR_PROGRAMS]
    programs = {}
    for n in ns.notes:
        key = "drum" if n.is_drum else str(n.program)
        programs[key] = programs.get(key, 0) + 1
    write_notes("mt3-guitar", seg, duration, load, infer, as_dict(guitar), programs=programs)
    write_notes("mt3-allinst", seg, duration, load, infer, as_dict(pitched), programs=programs)
    print(f"{seg}: {len(ns.notes)} notes ({len(guitar)} guitar), programs {programs}, "
          f"{infer:.1f}s for {duration:.1f}s audio, load {load:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1])
