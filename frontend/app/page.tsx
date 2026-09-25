"use client";

import { useState } from "react";
import UploadForm from "@/app/components/UploadForm";
import JobStatus from "@/app/components/JobStatus";

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-4xl flex-col gap-8 px-4 py-12 sm:px-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Riff<span className="text-indigo-600">scribe</span>
        </h1>
        <p className="text-slate-600">Upload a recording and get guitar tablature back.</p>
      </header>

      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
          New transcription
        </h2>
        <UploadForm onJobCreated={setJobId} />
      </section>

      {jobId && <JobStatus key={jobId} jobId={jobId} />}
    </main>
  );
}
