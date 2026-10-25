-- Metadata lives in PostgreSQL; model weights and recordings stay on disk.
CREATE TABLE IF NOT EXISTS audio_studio.workspaces (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE TABLE IF NOT EXISTS audio_studio.models (
    workspace_id text NOT NULL REFERENCES audio_studio.workspaces(id) ON DELETE CASCADE,
    id text NOT NULL,
    name text NOT NULL,
    engine text NOT NULL CHECK (engine IN ('kokoro', 'f5')),
    path text NOT NULL DEFAULT '',
    builtin boolean NOT NULL DEFAULT false,
    position bigint GENERATED ALWAYS AS IDENTITY,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    PRIMARY KEY (workspace_id, id)
);

CREATE TABLE IF NOT EXISTS audio_studio.presets (
    workspace_id text NOT NULL REFERENCES audio_studio.workspaces(id) ON DELETE CASCADE,
    id text NOT NULL,
    name text NOT NULL,
    model_id text NOT NULL,
    engine text NOT NULL CHECK (engine IN ('kokoro', 'f5')),
    category text NOT NULL DEFAULT 'custom',
    language text NOT NULL DEFAULT 'z',
    voice text NOT NULL DEFAULT '',
    speed double precision NOT NULL DEFAULT 1 CHECK (speed BETWEEN 0.5 AND 2),
    builtin boolean NOT NULL DEFAULT false,
    position bigint GENERATED ALWAYS AS IDENTITY,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    PRIMARY KEY (workspace_id, id),
    FOREIGN KEY (workspace_id, model_id) REFERENCES audio_studio.models(workspace_id, id)
);

CREATE TABLE IF NOT EXISTS audio_studio.jobs (
    workspace_id text NOT NULL REFERENCES audio_studio.workspaces(id) ON DELETE CASCADE,
    id text NOT NULL,
    kind text NOT NULL,
    status text NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    name text,
    model_id text,
    error text,
    request jsonb,
    result jsonb,
    snapshot jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(snapshot) = 'object'),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    PRIMARY KEY (workspace_id, id),
    FOREIGN KEY (workspace_id, model_id) REFERENCES audio_studio.models(workspace_id, id)
);

CREATE TABLE IF NOT EXISTS audio_studio.audio_assets (
    workspace_id text NOT NULL REFERENCES audio_studio.workspaces(id) ON DELETE CASCADE,
    id text NOT NULL,
    path text NOT NULL,
    filename text NOT NULL,
    name text NOT NULL,
    text text NOT NULL DEFAULT '',
    model_id text,
    preset_id text,
    job_id text,
    duration_seconds double precision CHECK (duration_seconds >= 0 AND duration_seconds < 'Infinity'::float8),
    sample_rate integer CHECK (sample_rate > 0),
    tags text[] NOT NULL DEFAULT ARRAY[]::text[],
    favorite boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    PRIMARY KEY (workspace_id, id),
    UNIQUE (workspace_id, path),
    FOREIGN KEY (workspace_id, model_id) REFERENCES audio_studio.models(workspace_id, id),
    FOREIGN KEY (workspace_id, preset_id) REFERENCES audio_studio.presets(workspace_id, id)
        ON DELETE SET NULL (preset_id),
    FOREIGN KEY (workspace_id, job_id) REFERENCES audio_studio.jobs(workspace_id, id)
);

CREATE TABLE IF NOT EXISTS audio_studio.datasets (
    workspace_id text NOT NULL REFERENCES audio_studio.workspaces(id) ON DELETE CASCADE,
    id text NOT NULL,
    name text NOT NULL DEFAULT '',
    csv_path text NOT NULL,
    wavs_dir text NOT NULL,
    valid boolean NOT NULL DEFAULT false,
    row_count integer NOT NULL DEFAULT 0 CHECK (row_count >= 0),
    sample_count integer NOT NULL DEFAULT 0 CHECK (sample_count >= 0),
    total_duration_seconds double precision NOT NULL DEFAULT 0
        CHECK (total_duration_seconds >= 0 AND total_duration_seconds < 'Infinity'::float8),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    PRIMARY KEY (workspace_id, id)
);

CREATE TABLE IF NOT EXISTS audio_studio.training_runs (
    workspace_id text NOT NULL REFERENCES audio_studio.workspaces(id) ON DELETE CASCADE,
    id text NOT NULL,
    name text NOT NULL DEFAULT '',
    dataset_id text,
    model_id text,
    job_id text,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    run_dir text NOT NULL DEFAULT '',
    config jsonb NOT NULL DEFAULT '{}'::jsonb,
    result jsonb,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    PRIMARY KEY (workspace_id, id),
    FOREIGN KEY (workspace_id, dataset_id) REFERENCES audio_studio.datasets(workspace_id, id),
    FOREIGN KEY (workspace_id, model_id) REFERENCES audio_studio.models(workspace_id, id),
    FOREIGN KEY (workspace_id, job_id) REFERENCES audio_studio.jobs(workspace_id, id)
);

CREATE INDEX IF NOT EXISTS jobs_workspace_created_idx ON audio_studio.jobs(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_workspace_status_idx ON audio_studio.jobs(workspace_id, status);
CREATE INDEX IF NOT EXISTS audio_workspace_created_idx ON audio_studio.audio_assets(workspace_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS audio_workspace_favorite_idx ON audio_studio.audio_assets(workspace_id, favorite);
CREATE INDEX IF NOT EXISTS audio_tags_idx ON audio_studio.audio_assets USING gin(tags);
CREATE INDEX IF NOT EXISTS training_workspace_status_idx ON audio_studio.training_runs(workspace_id, status);
