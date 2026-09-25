/** @type {import('next').NextConfig} */
const nextConfig = {
  // Emits .next/standalone: a minimal server.js plus only the node_modules it
  // needs, so the Docker runtime image doesn't carry the full dependency tree.
  output: "standalone",
};

export default nextConfig;
