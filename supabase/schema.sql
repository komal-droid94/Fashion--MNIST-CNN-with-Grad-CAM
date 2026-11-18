-- Fashion-MNIST CNN: Supabase schema.
-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run. Safe to re-run.
--
-- Security model
--   * The Streamlit app uses the public ANON key. It can INSERT predictions and READ the
--     recent_predictions view and the runs table. It cannot read raw prediction rows
--     (no image hashes leave the database) and cannot write runs.
--   * train.py uses the SERVICE key (kept in a local .env file, never committed) to record runs.
--   * No images are stored anywhere: only a SHA-256 hash, the predicted class and probabilities.

create table if not exists public.runs (
  id          bigint generated always as identity primary key,
  created_at  timestamptz not null default now(),
  epochs      int    not null,
  batch_size  int    not null,
  lr          real   not null,
  seed        int    not null,
  augment     boolean not null,
  params      int    not null,
  val_acc     real   not null check (val_acc between 0 and 1),
  test_acc    real   not null check (test_acc between 0 and 1),
  device      text
);

create table if not exists public.predictions (
  id               bigint generated always as identity primary key,
  created_at       timestamptz not null default now(),
  image_sha256     text   not null check (char_length(image_sha256) = 64),
  predicted_class  text   not null,
  confidence       real   not null check (confidence between 0 and 1),
  probabilities    jsonb  not null,
  source           text   not null check (source in ('upload', 'sample')),
  auto_inverted    boolean not null default false,
  model_test_acc   real
);

-- Explicit table privileges (row-level security below narrows them further).
grant select on public.runs to anon, authenticated;
grant insert on public.predictions to anon;

alter table public.runs        enable row level security;
alter table public.predictions enable row level security;

-- Anyone may read run summaries.
drop policy if exists "runs are public" on public.runs;
create policy "runs are public" on public.runs for select to anon, authenticated using (true);

-- The app (anon key) may add predictions, with sane values only. No anon SELECT policy,
-- so raw rows (including hashes) are not readable with the anon key.
drop policy if exists "app can log predictions" on public.predictions;
create policy "app can log predictions" on public.predictions for insert to anon
  with check (confidence between 0 and 1 and source in ('upload', 'sample'));

-- Public, hash-free view of the latest predictions for the "Recent predictions" tab.
create or replace view public.recent_predictions as
  select id, created_at, predicted_class, confidence, source, auto_inverted
  from public.predictions
  order by created_at desc
  limit 200;

grant select on public.recent_predictions to anon, authenticated;
