"use client";

import { useState } from "react";
import UploadForm from "@/app/components/UploadForm";
import JobStatus from "@/app/components/JobStatus";

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);

  return (
    <main className="flex min-h-screen flex-col items-center gap-8 p-8">
      <h1 className="text-2xl font-bold">StratoTab</h1>
      <UploadForm onJobCreated={setJobId} />
      {jobId && <JobStatus key={jobId} jobId={jobId} />}
    </main>
  );
}
