ALTER TABLE user_skills
  ADD COLUMN IF NOT EXISTS ai_extracted_skills TEXT[] NULL;

CREATE INDEX IF NOT EXISTS user_skills_ai_extracted_skills_gin_idx
  ON user_skills USING GIN (ai_extracted_skills);
