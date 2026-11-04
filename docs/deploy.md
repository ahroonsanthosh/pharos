# Putting Pharos online

The public demo runs as a free [Hugging Face Space](https://huggingface.co/spaces)
(2 CPUs, 16 GB of memory; it sleeps after two days without visitors and wakes on the
next visit). Every push to `main` updates it through
[.github/workflows/deploy-space.yml](../.github/workflows/deploy-space.yml).

## One-time setup

1. Make a free account at huggingface.co.
2. Create a token under Settings, Access Tokens, with the **Write** role.
3. In the GitHub repository, Settings, Secrets and variables, Actions:
   * add a **secret** `HF_TOKEN` with that token;
   * add a **variable** `HF_SPACE` with the Space to publish to, such as
     `your-hf-name/pharos` (the workflow creates it if it does not exist).
4. Run the **Deploy to Hugging Face** workflow from the Actions tab, or push to `main`. The first
   build takes a few minutes; the Space is then at
   `https://huggingface.co/spaces/your-hf-name/pharos`.

## Public mode

The Space sets `PHAROS_PUBLIC=1`, because anyone can reach it. Visitors can run the
built-in demo flights, their own browser camera, and footage they upload (DJI video
with its `.SRT`, or photos), on maps they upload. The server will not open paths on
itself, fetch stream URLs, use attached cameras or listen for MAVLink. A file may be
up to 500 MB and all uploads together 4 GB; two feeds run at once, each for at most
15 minutes. Everyone sees the same feeds, and uploads are lost when the Space
restarts.

For live drone feeds (RTMP from a DJI controller, MAVLink from a ground station),
run Pharos on your own laptop: see [laptop.md](laptop.md).
