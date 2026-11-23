# transcribe

Made by Animato for your account. This private repo turns your voiceover into text with word timings, using GitHub Actions,
so your phone does not have to. The speech model is downloaded once and kept in the Actions cache; the app wakes it up every time you open it.
Each job lives on a short-lived branch called `job-<id>` that the app deletes when it is done.

You can delete this repo at any time; the app will make it again the next time you sync a voiceover.
