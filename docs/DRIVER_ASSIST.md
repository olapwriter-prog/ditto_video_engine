# Driver assist from two or more phones: assessment and plan

Status: **proposal.** Written 2026-11-10, in answer to the idea of reusing this engine's
machine-learning parts for a driver-assist program that links several phones in one
vehicle and tells the driver when to change lanes, slow down and so on. The only thing
built is a convolution kernel and a cost benchmark (section 5).

Speed figures are estimates unless marked as measured. No camera, phone or road has been
tested.

## 0. Requirements set by the owner (2026-11-10)

1. **Warnings only.** The phone never controls the vehicle. (Section 2.)
2. **India and all European countries.** (Section 2a.)
3. **Any phone: Android, iPhone, or agenticOS.** "mac" in the request is read as Apple's
   phones. (Section 2b.)

---

## 1. Short answer

It is feasible as a **warning system**, and about half of what it needs is already planned
for Ditto. It is not feasible, and should not be attempted, as anything that steers, brakes
or accelerates.

It is also close to the opposite of Ditto in how it runs:

| | Ditto (images and video) | Driver assist |
|---|---|---|
| Direction | Generates pictures | Understands pictures |
| When | At night, on the charger, nobody waiting | While driving, someone's safety depending on it |
| Deadline | Hours are fine | About 100 ms per frame, every frame |
| A slow or wrong result | A late or poor clip | A missed or false warning |
| If interrupted | Resume tomorrow | Must say so at once and fall silent |

So it shares kernels with Ditto, not a pipeline. It belongs in its own crate (or its own
repository) on top of the same `rie-core` and the shared operations in `ditto-core`.

## 2. Scope: what it does and what it never does

**It advises. It never controls the vehicle.** There is no connection to steering, brakes
or throttle, and none is planned. The driver stays responsible at all times.

In scope, in the order they should be built:

| Function | Needs | Says |
|---|---|---|
| Forward collision warning | Front camera, own speed | "Slow down", when the time to reach the vehicle ahead falls below a threshold |
| Following distance | Front camera, own speed | "Too close" |
| Lane departure | Front camera | "Drifting left / right", when crossing a marking without indicating intent |
| Lane-change advice | **Rear or side phone** | "Clear on the right" / "Vehicle on your right, wait" |
| Speed advice | GPS speed, sign reading or map data | "Limit is 60" |
| Driver attention (optional) | A phone facing the driver | "Eyes on the road" |

Out of scope: any actuation, automatic lane keeping, automatic braking, navigation,
recording or uploading of video, and any claim that the system makes driving safe.

Three rules follow from "someone's safety depends on it":

1. **Silence is never read as "all clear".** When the system cannot see (glare, night,
   heat, a phone dropped off the link), it says once that it is not watching and stops.
2. **Voice and tones only.** Nothing to read or touch while moving. This also fits
   agenticOS, which has no screen to look at.
3. **Too many false warnings is a failure**, not caution: a driver who learns to ignore it
   is worse off than one without it.

## 2a. India and all of Europe: what that requires

One program, with everything that differs between countries held as data, chosen from the
GPS position at the start of a drive and changeable by voice.

| Differs by country | Cases to cover | Handled by |
|---|---|---|
| Side of the road | Left: India, UK, Ireland, Cyprus, Malta. Right: the rest of Europe | A flag that mirrors lane logic: which side is the overtaking lane, which mirror matters, which way "drifting" is dangerous |
| Speed units | km/h everywhere except mph in the UK | Country table; speech in the local unit |
| Speed limits and signs | European signs are broadly uniform; national defaults differ; Indian signs and their upkeep differ | Sign reading where signs are reliable, a table of national default limits per road type otherwise, and saying "limit unknown" instead of guessing |
| Lane markings | Europe: mostly marked. India: often unmarked, worn, or ignored | Lane functions switch on only where markings are detected with confidence. Elsewhere the system warns on **vehicles and gaps**, not lanes |
| Traffic mix | India: two-wheelers, three-wheelers, pedestrians, animals, wrong-side traffic. Europe: cyclists, trams, roundabouts | Detector classes and condition experts (section 3) per region; evaluated per region |
| Language | Hindi, Telugu and the other major Indian languages; some 24 official languages in the EU | Short fixed phrases, translated once and reviewed by speakers, spoken by the platform's own voice. Tones as the fallback for any language not yet covered |
| Law | Rules on mounting a phone, using one while driving and filming the road differ by country; some European countries are reported to restrict dashcams | A per-country checklist **before release in that country**. The design helps: no video is stored or sent |
| Privacy (EU and India) | Filming people and number plates on public roads | Frames are processed in memory and dropped. The overnight-learning buffer (section 6) is off by default and is opt-in per country |

"All countries" is the target; it is reached country by country. A country counts as
supported only after the system has been evaluated on footage from it and the legal
checklist is done. The honest launch order is India plus one right-hand and one left-hand
European country, then the rest.

## 2b. Any phone: Android, iPhone, agenticOS

One Rust core, three thin shells. Only capture, voice and the app wrapper differ.

| Layer | Android | iPhone | agenticOS (postmarketOS) |
|---|---|---|---|
| Core: models, tracking, fusion, warning logic | Same Rust library | Same | Same |
| How the core is called | JNI, as RIE's `bindings/android` | C interface from Swift, as RIE's `bindings/ios` | Direct, a child process of `agentd` |
| Camera | Platform camera API | Platform camera API | Kernel camera interface; **works on few phones today** |
| GPU | Vulkan | Metal | Vulkan through Mesa, unproven |
| Voice | Platform text-to-speech | Platform text-to-speech | RIE's speech path |
| Position and speed | Platform location API | Platform location API | `agentd`'s location tool |
| Must stay in the foreground | Can run as a foreground service | The camera stops when the app leaves the screen | Not applicable |
| Distribution | App store or direct install | App store only, with review | agenticOS package |

What "any phone" can honestly mean:

- **Any phone can join; not any phone can do everything.** At start-up each phone runs a
  ten-second self-test (the benchmark in section 5) and takes the role it can hold: full
  perception, a smaller model at a lower frame rate, or sensor-only (GPS, motion) with no
  camera work. A phone that cannot hold the minimum frame rate says so and does not
  pretend to watch.
- **Mixed sets must work**: an iPhone in front with an Android phone behind. So the link
  uses only what all three have: a local Wi-Fi network (one phone's hotspot) with a small
  common message format. No platform-specific pairing.
- agenticOS is the hardest of the three for this product, because of cameras. Android and
  iPhone are where it will work first.

## 3. What carries over from this engine

| Part | Use here | State |
|---|---|---|
| `rie-core` quantized kernels, thread pool, core detection | Every model below | Exists |
| 4-bit / 8-bit weights, GGUF files, the converter | Small, fast perception models | Exists for LLMs; converter planned (Ditto Phase 1) |
| Convolution kernels (2-D), group/layer norm | Detection and lane models are mostly convolutional | Planned for Ditto's image decoder (Phase 1). **Shared work** |
| Full attention kernel (`ditto-core`) | Transformer detectors | Built, x86 only so far |
| Experts chosen from a table (plan level B) | A set of experts per driving condition: day, night, rain, glare. Chosen once every few seconds, no per-frame cost | Planned |
| Skip expert (plan level D) | Consecutive frames are nearly the same: recompute only the parts of the picture that changed | Planned. Worth more here than in generation |
| GPU backend (`rie-gpu`) | Probably required to hold the frame rate | Exists for LLMs; Mesa drivers unproven |
| Speech in RIE (Telugu, Hindi, numbers spoken as words) | The warnings themselves | Exists |
| Night runner, checkpoints, duty cycle | Not for driving. Useful for one thing: **learning overnight from the day's recorded drive** (section 6) | Planned (Phase 3) |

New work that Ditto does not need: camera capture, calibration, tracking over time,
distance and closing-speed estimation, the link between phones, the warning logic, and a
real-time scheduler with a deadline per frame.

## 4. Why more than one phone

One phone on the windscreen sees forward only. Advice about **changing lanes** needs to
know what is behind and beside the vehicle, which is what the second phone is for.

| Phones | Placement | Adds |
|---|---|---|
| 1 | Windscreen, facing forward | Collision, distance, lane departure, signs |
| 2 | + rear window, facing back | Vehicles approaching from behind and in adjacent lanes: lane-change advice |
| 3 | + facing the driver, or a second rear corner | Driver attention, or a wider rear view |

How they work together:

- **Each phone runs its own models and sends results, not video**: a few hundred bytes per
  frame (tracked objects with position, closing speed, confidence; lane geometry; a
  heartbeat). This keeps the link cheap and keeps video on the phone that took it.
- **One phone is the hub**: it merges the tracks into one picture of the surroundings,
  decides, and speaks. The others are sensors.
- **Link**: the hub's Wi-Fi hotspot, or USB between phones. Local only; no internet needed.
- **Clocks**: frames carry timestamps from a clock synchronised over the link. At 100 km/h
  a vehicle moves 2.8 m in 100 ms, so an unsynchronised rear phone gives wrong gaps.
- **Calibration**: each phone needs its height, tilt and heading relative to the vehicle.
  Estimated at the start of a drive from the horizon and lane lines, checked continuously,
  and the system refuses to give distances until it is calibrated.
- **Losing a phone** degrades the functions that depended on it, announced once.

## 5. Models and the time budget

Small existing models, as with Ditto. No training from scratch. Licences must be checked
before any is chosen; some popular detectors are under licences that restrict commercial
or closed use.

| Task | Kind of model | Rough cost per frame |
|---|---|---|
| Vehicles, people, two-wheelers, animals | Small single-stage detector at ~416×416 | 1 to 4 G multiply-adds |
| Lane markings and road edge | Row-wise lane model or light segmentation | 1 to 10 G |
| Distance | Geometry from the calibrated camera and object size; optional small depth model | Negligible, or 5 G+ |
| Tracking, closing speed | Classical tracker (no neural network) | Negligible |
| Speed signs | Detector crop + small classifier, a few times a second | Small |

Budget: Ditto's plan assumes about 15 G multiply-adds per second on a Snapdragon 845
class CPU, **unmeasured**. At that rate a detector plus a lane model fits at roughly 3 to
7 frames per second on the CPU. Warnings need about 10. So either the models must be at
the small end, or the skip expert and the GPU must deliver. This has to be measured before
anything is promised.

**Built for this:** a 2-D convolution on `rie-core`'s quantized kernels (also needed by
Ditto's image decoder) and `ditto bench-conv`, which times a stack of convolutions the size
of a small detector's backbone with random weights. It measures cost only; it detects
nothing. Measured on the x86-64 development VM (4 cores, AVX2, 4-bit weights), **not a
phone**:

| Input | Work per frame | Frames per second |
|---|---|---|
| 416×416, full width | 3.2 G multiply-adds | 11 (4.6 on one thread) |
| 416×416, half the channels | 0.9 G | 23 |
| 640×640, full width | 7.6 G | 5 |

Convolutions run at 36 G multiply-adds per second here, against 57 G/s for plain matrix
products: laying out the patches costs a third. If a phone reaches a third of this VM, as
the plan assumes, one full-width model runs at about 4 frames per second on the CPU and a
half-width one at about 8. `scripts/phone-bench.sh` now runs these cases on the phone.

End-to-end target: under 300 ms from an event in front of the camera to the start of the
spoken warning, measured, with the worst case reported and not the average.

## 6. Learning overnight

This is where the "night" design fits. During a drive the phones keep a short, local,
encrypted buffer of moments where the system was unsure or the driver braked hard without
a warning. At night, on the charger, in agenticOS's Deep stage:

- those moments are re-examined with a larger, slower model than can run while driving,
- its answers become training examples for the small models (the same overnight LoRA
  mechanism agenticOS already uses for its language model, kept only if it scores better
  on held-out moments),
- the buffer is then deleted. Nothing is uploaded.

## 7. Risks specific to this idea

| Risk | Why it matters | Answer |
|---|---|---|
| **Cameras on postmarketOS** | On many supported phones the mainline kernel drives the camera poorly or not at all | Android and iPhone first; agenticOS per phone, where the camera works |
| **"Any phone"** | Old and low-end phones cannot hold the frame rate; iPhones stop the camera in the background | Start-up self-test and roles (section 2b); a stated minimum below which a phone is sensor-only |
| **Three platforms** | Three shells to build, test and release, two of them through app-store review | One Rust core with tests that need no phone; shells kept thin |
| **Thirty-odd countries** | Each has its own roads, signs, language and law | Country data in tables; a country is supported only after evaluation and the legal check (section 2a) |
| **Heat** | A phone on a dashboard in the sun, charging, running flat out, will throttle or shut down | Measure sustained frame rate at 45 °C ambient. Design for the throttled rate |
| **Roads the models were not trained on** | Public lane and detection models are mostly trained on marked highways abroad. Unmarked lanes, mixed traffic, autos, cattle and left-hand driving are different | Evaluate on local footage before anything else; use Indian driving datasets where licences allow; expect lane functions to be limited to marked roads |
| **Night, rain, glare** | Phone cameras degrade badly | Condition experts; announce "not watching" rather than guess |
| **False confidence** | A driver may lean on advice that is wrong | Advisory wording, conservative lane-change advice ("wait" when unsure, never "go" on thin evidence), clear limits stated at start |
| **Law** | Rules on phones in vehicles, mounting and recording differ by country and state | Check for the target market before any road test |
| **Testing** | Bugs cannot be found safely by driving at them | Recorded drives first, then a passenger operating it, never the driver |
| **Competes with Ditto for time** | Same people, same kernels | Sequence after Ditto's Phase 1, which delivers the shared convolution kernels and converter |

## 8. Phases

| Phase | Work | Done when |
|---|---|---|
| **D0. Can it see** | `ditto bench-conv` on one Android phone, one iPhone and one agenticOS phone (the convolution kernel and benchmark exist; ARM is untested). Camera capture, frame rate and delay on each. Temperature on a dashboard. | A measured frames-per-second budget per platform, and which agenticOS phones have a usable camera |
| **D1. Offline, one camera** | Converter for one detector and one lane model; run them on recorded drives on a desktop; tracking, calibration, distance, time to collision. | Detection and warning accuracy reported on recorded local footage, with missed and false warnings counted |
| **D2. Real time, one phone** | Android shell first, then iPhone. Deadline scheduler, skip expert, voice warnings, "not watching" states, start-up self-test and roles. | Holds the frame rate and the 300 ms target on the phone for an hour, on recorded video fed in as if live |
| **D3. Two phones** | Link over local Wi-Fi, clock sync, hub merging, rear tracking, lane-change advice, left- and right-hand traffic. | Lane-change advice correct on recorded two-camera drives in both traffic directions; a mixed Android and iPhone pair works; a dropped phone is handled |
| **D4. Road trials** | Passenger-operated drives; tune thresholds; count false warnings per hour. | A false-warning rate low enough that test drivers keep it on |
| **D4b. Countries** | Country tables, phrases and voices, legal checklist, evaluation on local footage, one country at a time. | India, one right-hand and one left-hand European country pass; then the rest |
| **D5. Overnight learning** | Buffer, larger night model, adapter training with the held-out gate. | Small models measurably better on local footage after a week |

D0 and D1 need no phone in a car and risk nobody. Nothing goes on a road before D2 passes.

## 9. Decisions

Settled (section 0): warnings only; India and all European countries; Android, iPhone and
agenticOS.

Still needed:

1. **Is this a second product next to Ditto, or does it replace video generation as the
   priority?** They share kernels but not much else.
2. **Which phones do you have to test on**, one per platform, and is there recorded
   driving footage (India and Europe) to start D1 with?
3. **Is "mac" the iPhone?** If you meant Mac computers, section 2b changes.
4. **The first three countries** for D4b. Proposed: India, Germany (right-hand), UK or
   Ireland (left-hand).
5. **Who publishes the apps.** The iPhone and Android stores need a developer account and
   an owner for the review process.
