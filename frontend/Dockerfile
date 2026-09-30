# Production build using Next.js standalone output (see next.config.mjs).

FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-slim AS builder
WORKDIR /app
# NEXT_PUBLIC_* values are inlined into the client JS bundle by `next build`,
# so they must be present HERE, as build args - setting them as runtime
# container env vars has no effect on an already-built bundle. Defaults match
# local dev; docker-compose.yml passes them explicitly, and a deploy target
# overrides them with --build-arg.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_ENABLE_URL_INGESTION=true
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_ENABLE_URL_INGESTION=$NEXT_PUBLIC_ENABLE_URL_INGESTION \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
# node:20-slim ships a non-root "node" user; no need to create one.
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
# public/ holds only the alphaTab browser files, written by `prebuild`
# (scripts/copy-alphatab.mjs) in the builder stage.
COPY --from=builder --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
