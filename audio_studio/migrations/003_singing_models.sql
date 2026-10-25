ALTER TABLE audio_studio.models DROP CONSTRAINT models_engine_check;
ALTER TABLE audio_studio.models ADD CONSTRAINT models_engine_check
    CHECK (engine IN ('kokoro', 'f5', 'rvc', 'diffsinger'));
