# Pictures and video: what the engine is for, beyond generating

Written 2026-11-10. This widens Ditto from "generate images and video at night" to "the
phone's engine for pictures and video": understanding the user's own pictures, turning
screenshots into SVG, and handing what it finds to the Rust inference engine (RIE), which
holds the language model, the memory and the answer pages.

**Status column:** *built* means code in this repository with tests; *code* means it needs
no model and is not written yet; *model* means it needs a model that is not chosen,
converted or measured yet. Nothing marked *model* works today.

The use cases are collected from what phone users commonly do with pictures. They were
written from general knowledge, not from a user study or a fresh survey; treat the list as
a starting catalogue to prune.

## 1. What is built now

### `ditto look`: what a picture contains, without a model

```
$ ditto look IMG_2041.jpg
IMG_2041.jpg: photo, 4032x3024 (12.2 MP), taken 2026-03-01 18:22:05 with Google Pixel 7
at 17.38500, 78.48670. Colours: green 41%, blue 22%. QR code: upi://pay?pa=shop@bank...
```

| Extracted | From | Notes |
|---|---|---|
| When, camera, exposure, orientation | The file's EXIF block | Absent facts are left out, never guessed |
| Where (latitude, longitude, altitude) | EXIF GPS | "0, 0" (no fix) is treated as no position |
| Size, format, megapixels | The file | png, jpeg, webp, gif, bmp |
| Photo or screenshot/graphic | Camera data present; share of flat areas | A measurement with a threshold, labelled as such |
| Blurry, very dark, overexposed | Variance of the Laplacian, brightness | Only reported for photographs |
| Main colours, by name and share | Pixel counts | |
| QR codes (links, UPI payments, Wi-Fi) | Decoded | Other barcodes: not yet |
| Fingerprint | 64-bit difference hash | Same picture resized or recompressed differs in few bits: finds duplicates |

It runs in milliseconds, so it runs when a picture arrives, not at night.

**It does not read text, name objects, recognise people or describe a scene.** Those need
models (section 3).

### `ditto trace`: screenshots, diagrams and logos as SVG

Reading of the request: people take a screen capture from a browser and want an SVG of
it, and the tools they try give results that are subtly wrong or need tuning. If you meant
a different SVG problem, tell me and this section changes.

How it works: the picture is reduced to a small palette (16 colours by default), and each
colour becomes one path along the pixel edges of its area. Rendering the SVG gives back
the reduced picture **exactly**, at any zoom. There are no smoothing parameters to get
wrong.

Checked two ways:

- In the tests, an independent even-odd fill of the outlines reproduces every pixel
  (rectangles, rings with islands, checkerboards, diagonals, transparency, noise).
- With a real browser: a page was captured in Chromium (420×300), traced, the SVG
  rendered again in Chromium, and that rendering traced back to a **byte-identical SVG**,
  at 8, 16 and 256 colours.

Limits, measured on that capture (a 13.5 KB PNG):

| Colours | SVG size | Note |
|---|---|---|
| 8 | 29.6 KB | Text edges visibly stepped in colour |
| 16 | 45.5 KB | Default |
| 256 | 67.5 KB | The command warns that the picture is not flat |

- **Anti-aliased text makes the SVG two to five times larger than the PNG.** The capture
  had 615 distinct colours, nearly all of them text edges. The command reports this
  instead of hiding it.
- Text stays outlines: it cannot be selected or edited. Real `<text>` elements need text
  reading (section 3), which would also shrink the file most.
- Edges are pixel staircases, which is what makes it exact. A smoothed mode (curves,
  smaller, no longer exact) is not written.
- Photographs are refused above 16 megapixels and flagged below.

### Handing results to the inference engine

Three paths, each tested against the inference engine's **own code** (its manifest loader
and tool runner from `rie-server`, its memory store from `rie-core`):

| Path | How | Use |
|---|---|---|
| The engine calls Ditto | `rie-server --headless-tools integrations/rie/ditto-tools.json` gives it two tools, `photo.inspect` and `image.to_svg`. Their output is a flat JSON object, which the engine's "ask the phone" page shows as a card of fields | "When and where was this photo taken?" |
| Ditto tells a running engine | `ditto look --rie http://127.0.0.1:8080 PICTURE` posts one sentence per picture to `/v1/memory` | Pictures become things the language model can recall later |
| Ditto writes the engine's memory directory | `ditto look --memory DIR PICTURE`, using the engine's store format through its own library | When no server is running, such as a night job. **Not while a server has the same directory open:** use `--rie` then |

Safeguards: observations are only sent to an address on the phone itself (they contain
where the user was); a picture already remembered is not stored twice; the SVG tool never
overwrites a file and writes only next to its input.

Not tested: a live `rie-server` with a real model answering a question through these tools.
The pieces on both sides of that are tested; the full loop needs a model file.

## 2. Use cases

### Understanding the user's own pictures

| Use case | Example | Needs | Status |
|---|---|---|---|
| Facts about a photo | "When and where did I take this?" | EXIF | **built** |
| Codes | Pay a UPI code, join Wi-Fi, open a link from a photo | QR decoding | **built** (QR); other barcodes: code |
| Clean up the gallery | Find duplicates, blurry and dark shots | Fingerprint, sharpness | **built** (measurements); a sweep command: code |
| Scan a document | Photo of a page to a straight, clean, flat scan | Edge finding, perspective correction, thresholding | code |
| Read text | Receipts, bills, business cards, signs, prescriptions' printed parts, whiteboards | Text detection and recognition, including Indian scripts | model |
| Text to structured data | Receipt to merchant, date, total; card to a contact; bill to a due date and reminder | Text reading, then RIE's language model with schema-constrained output (which it already has) | model |
| Meter and odometer readings | Electricity or water meter photo to a number in a log | Digit reading | model |
| Translate what the camera sees | Signs and menus while travelling | Text reading, then RIE | model |
| Search by meaning | "photos of the dog at the beach" | Image embeddings stored in RIE's 4-bit vector index (which exists) | model |
| Describe aloud | A picture described in Telugu or Hindi for someone who cannot see it | Captioning model, RIE's speech output | model |
| Group by person | "photos with Amma" | Face detection and embedding. On the device only, off by default, the user's own library only | model |
| Before sharing | Blur faces and number plates, remove location | Detection; agenticOS's `photos` crate already strips metadata | model + code |

### Screenshots

| Use case | Example | Needs | Status |
|---|---|---|---|
| Screenshot to SVG | A diagram or UI capture as a scalable file | Tracing | **built**, exact, large when there is text |
| Compact, editable SVG | Text as real text, shapes as shapes | Text reading, shape fitting | model + code |
| What is on this screen | The agent reads a screenshot: text, buttons, links | Text reading, layout analysis | model |
| Chart to numbers | A bar chart in a screenshot back to a table | Text reading, geometry | model + code |
| Remember screenshots | Tickets, order confirmations, addresses found again by asking | Text reading, RIE memory (the hand-over exists) | model |

### Editing existing pictures

| Use case | Example | Needs | Status |
|---|---|---|---|
| Crop to a specification | Passport or ID photo size | Face position, geometry | model (small) + code |
| Collage, poster, greeting card | Festival card with a photo and text in any script | Layout and text drawn by code: diffusion models garble writing | code |
| Remove background, erase an object | Product photo on white; remove a passer-by | Segmentation, inpainting | model, night |
| Upscale, denoise, low light | Old or small photos made usable | Restoration model | model, night |
| Restore and colourise | Old family prints | Diffusion | model, night |

### Video

| Use case | Example | Needs | Status |
|---|---|---|---|
| Summarise a clip | Key frames and a contact sheet | Scene-change detection | code, once video can be decoded |
| Find a moment | "where the cake is cut" | Frame embeddings | model |
| Transcribe and subtitle | Speech in a clip to text and captions | Speech model on the RIE side | model |
| Stabilise, timelapse, slow motion | Shaky clip steadied; frames interpolated | Motion estimation; interpolation model | code / model |
| Photo to short clip | A still photo brought to life | Image-to-video (plan, Phase 8) | model, night |
| Trip or event film | Chosen photos and clips cut into a short film with titles | Selection by quality and embeddings; editing by code | model + code |
| Watch with a spare phone | Motion or a person at the door, as an alert | Motion detection; detector (shared with the driver proposal) | code / model |
| Driver warnings | See [DRIVER_ASSIST.md](DRIVER_ASSIST.md) | | proposal |

**An open problem for all video input:** decoding phone video (H.264, HEVC) needs a
decoder. This engine has none. Options are the platform's decoder on Android and iPhone,
and on agenticOS a hardware decoder through the kernel or a packaged library. Undecided.

### Generation

Images and clips from text, in [PLAN.md](PLAN.md).

### Not built, on purpose

- Identifying strangers from their face, or tracking a person across pictures that are not
  the user's own library.
- Putting a real person's face into pictures or video they were not in.
- Medical conclusions from photos of skin, eyes, scans or reports. Reading the printed
  text of a report is in scope; interpreting it as a diagnosis is not.
- The content limits already listed in the plan for generation.

## 3. The models this needs, and the order

Everything marked *model* above comes down to four small models, all of which fit the
kernels that exist (convolution, attention, quantized matrices):

| Model | Unlocks | Runs |
|---|---|---|
| 1. Text detection + recognition, Latin and Indian scripts | Receipts, cards, signs, meters, screenshots, compact SVG, translation. **The most use cases by far** | On arrival, seconds |
| 2. Image embedding (vision-language encoder) | Search by meaning, find a moment, pick the best shots | On arrival or in the night sweep |
| 3. Small detector (faces, plates, people, vehicles) | Redaction, crops, spare-phone alerts, driver warnings | On arrival; real time for driving |
| 4. Captioning | Describe aloud | Night, or seconds on a strong phone |

Proposed order: text reading first. It is the single largest gap between what `ditto look`
does today and "extract any information possible", and its output is exactly what RIE's
language model is good at structuring.

Each needs the converter from the plan's Phase 1, a licence check, and a measured accuracy
on real pictures (including Telugu and Hindi print) before it is called working.

## 4. Division of work between Ditto and RIE

| Ditto (pixels) | RIE (language) |
|---|---|
| Decode, measure, read codes, trace | Decide which tool a question needs |
| Run vision models, produce text and embeddings | Turn recognised text into structured records under a schema |
| Generate and edit pictures and clips at night | Remember observations and recall them into a conversation |
| | Answer pages, speech, Indian languages |

RIE to Ditto, later: the language model expands a short request into a full prompt for
generation, and chooses which pictures a film or collage should use.
