# Driver assist from two or more phones: assessment and plan

Status: **proposal only, nothing built.** Written 2026-11-10, in answer to the idea of
reusing this engine's machine-learning parts for a driver-assist program that links several
phones in one vehicle and tells the driver when to change lanes, slow down and so on.

Every speed figure here is an estimate. No camera, phone or road has been tested.

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
anything is promised; `ditto bench` gains a convolution case for that purpose.

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
| **Cameras on postmarketOS** | On many supported phones the mainline kernel drives the camera poorly or not at all | Check first, per phone. This alone can rule a phone out. Android via RIE's existing Android bindings is the fallback |
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
| **D0. Can it see** | Camera capture on the target phone under postmarketOS; frame rate and delay; temperature on a dashboard. Convolution case in `ditto bench`. | A go / no-go for the phone and a measured frames-per-second budget |
| **D1. Offline, one camera** | Converter for one detector and one lane model; run them on recorded drives on a desktop; tracking, calibration, distance, time to collision. | Detection and warning accuracy reported on recorded local footage, with missed and false warnings counted |
| **D2. Real time, one phone** | Deadline scheduler, skip expert, voice warnings, "not watching" states. | Holds the frame rate and the 300 ms target on the phone for an hour, on recorded video fed in as if live |
| **D3. Two phones** | Link, clock sync, hub merging, rear tracking, lane-change advice. | Lane-change advice correct on recorded two-camera drives; a dropped phone is handled |
| **D4. Road trials** | Passenger-operated drives; tune thresholds; count false warnings per hour. | A false-warning rate low enough that test drivers keep it on |
| **D5. Overnight learning** | Buffer, larger night model, adapter training with the held-out gate. | Small models measurably better on local footage after a week |

D0 and D1 need no phone in a car and risk nobody. Nothing goes on a road before D2 passes.

## 9. Decisions needed

1. **Is this a second product next to Ditto, or does it replace video generation as the
   priority?** They share kernels but not much else.
2. **Which country and which roads** (marked highways, city traffic)? This decides the
   data, the lane functions that are realistic, and the law to check.
3. **agenticOS phones only, or Android as well?** Camera support may force Android.
4. **Which phones**, and is there recorded driving footage to start D1 with?
5. **Confirmation of the scope in section 2**: warnings only, no control of the vehicle.
