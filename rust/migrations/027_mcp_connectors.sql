CREATE TABLE IF NOT EXISTS mcp_connectors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    publisher TEXT,
    version TEXT NOT NULL,
    homepage_url TEXT,
    repository_url TEXT,
    config_schema TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_connectors_name
    ON mcp_connectors(name);

CREATE INDEX IF NOT EXISTS idx_mcp_connectors_publisher
    ON mcp_connectors(publisher);
