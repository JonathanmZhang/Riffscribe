// Developer-only URL flags, read on the client:
//   ?debug=1           show the playback lag overlay (components/DebugOverlay)
//   &job=<job id>      open an existing job instead of submitting a new one
//   &src=stem          play the separated guitar stem instead of the mix
// Without debug=1 the other flags are ignored.
export interface DebugParams {
  debug: boolean;
  job: string | null;
  src: "audio" | "stem";
}

export function readDebugParams(): DebugParams {
  if (typeof window === "undefined") return { debug: false, job: null, src: "audio" };
  const params = new URLSearchParams(window.location.search);
  const debug = params.get("debug") === "1";
  return {
    debug,
    job: debug ? params.get("job") : null,
    src: debug && params.get("src") === "stem" ? "stem" : "audio",
  };
}
