# Revision — findings from real Omani cards

**Date:** 2026-08-24
**Status:** approved, supersedes parts of `2026-08-24-id-card-extraction-design.md`
**Evidence:** two real cards run through the deployed demo (citizen card and resident card)

---

## 1. What the real cards actually print

| Printed on the card | Citizen (البطاقة الشخصية) | Resident (بطاقة مقيم) |
|---|---|---|
| Civil number | ✅ 8 digits | ✅ 8 digits |
| Expiry date | ✅ DD/MM/YYYY | ✅ DD/MM/YYYY |
| Date of birth | ✅ DD/MM/YYYY | ✅ DD/MM/YYYY |
| مكان الميلاد (place of birth) | ✅ الامارات | ✅ جمهورية مصر العربية |
| الإسم (name) | ✅ **Arabic only** | ✅ **Arabic only** |
| المهنة (occupation) | ❌ | ✅ |
| **Nationality** | ❌ **not printed** | ❌ **not printed** |
| **Sex** | ❌ **not printed** | ❌ **not printed** |

## 2. Assumptions the evidence invalidated

**D1/D2 were backwards.** The original design assumed a Latin primary value with an
optional Arabic companion. Reality: the holder's name is printed **only in Arabic**.
There is no Latin name to extract, which is why the demo rendered Arabic text into the
Latin `full_name` slot.

**`nationality` and `sex` do not exist on these cards.** The model produced both by
inference — `Oman` from the "SULTANATE OF OMAN" header, and sex from the photo or from
the بنت / بن particle in the name. On the resident card the holder is Egyptian (place of
birth: جمهورية مصر العربية) and `nationality` still returned `Oman`.

**One of these reached the user as `ok`.** On the resident card, `sex` returned `M` with
status `ok` — a value that was never on the card, presented as read. This is precisely
the silent error the whole design exists to prevent. `nationality` escaped the same fate
only by accident: the model wrote `Oman`, the closed list contains `OMANI`, and the
format rule rejected the mismatch.

**The lesson is not that the status system failed.** It worked — it caught the non-ISO
dates and the bad nationality string. It was pointed at fields that cannot be read,
and no confidence mechanism can rescue a field that is not there.

## 3. Accuracy defects observed

| Defect | Evidence | Cause |
|---|---|---|
| Day/month swapped | Card 1 DOB `11/08/1989` → `1989-11-08` (8 Nov, should be 11 Aug) | Prompt never states the cards use DD/MM/YYYY |
| Dates not normalised | Card 2 → `29/09/2002`, `25/01/2027` returned raw | Same; format rule caught it (`review`) |
| Name truncated | Card 2 `زياد نشأت عبد الحى ابو الوفا حموده` → `زياد نشأت عبد الحى` | `MAX_NEW_TOKENS=256` across nine keys; long Arabic names hit the ceiling |
| Name misread | Card 1 `عهود` → `عهد` | Model misread; both passes disagreed, correctly flagged `review` |

## 4. Latency

Measured **14.1s, 17.1s, 19.1s**. The design predicted ~3.2s.

**The prediction was arithmetically wrong.** §15 of the original spec computed
"190 tokens @ ~30 tok/s → 2.4s"; 190 ÷ 30 is 6.3s. On top of that, realistic T4 decode
for a 3B VLM is nearer 15–20 tok/s than 30, and Arabic tokenises expensively — a
seven-component Arabic name is a large number of tokens.

Nine keys, three of them Arabic duplicates, is simply too much output.

## 5. Decisions

| # | Decision | Rationale |
|---|---|---|
| R1 | **Extract only what is printed.** Drop `nationality` and `sex` entirely. | They are not on the card. No status mechanism can make an invented value safe. |
| R2 | **Add `card_type`** (`citizen` \| `resident`). | It IS printed, it distinguishes the two card families, and it is what a reviewer needs to know first. |
| R3 | **Add `place_of_birth`** (Arabic). | Printed on both cards, and the honest proxy for the nationality question a bank actually asks. |
| R4 | **Name is Arabic-only.** `full_name_ar`, no Latin counterpart. | There is no Latin name on the card. |
| R5 | **Prompt states dates are DD/MM/YYYY** and must be converted to ISO. | Fixes the day/month swap, which produced a wrong-but-plausible date. |
| R6 | **Raise `MAX_NEW_TOKENS` to 320** and demand every name component. | Fixes truncation; affordable now that three redundant keys are gone. |
| R7 | **`SHOW_BOXES` defaults to false**; grounding pass not issued. | Boxes returned nothing usable in the live run. Per D4, drop rather than draw wrong ones. |
| R8 | **Stream the primary pass** to the UI. | Fields appear progressively instead of a blank multi-second wait. Perceived latency falls further than measured latency. |

**Self-consistency is retained.** It is the only mechanism that catches hallucination,
and this revision was caused by a hallucination. Removing it to buy speed, immediately
after it proved necessary, would be exactly the wrong trade.

## 6. Revised field set

```
card_type        "citizen" | "resident"          (Latin enum)
full_name_ar     Arabic, every component
id_number        8 digits
date_of_birth    ISO, converted from DD/MM/YYYY
expiry_date      ISO, converted from DD/MM/YYYY
place_of_birth_ar Arabic
```

Six fields, one long Arabic value instead of three. Estimated decode roughly halves.

## 7. Revised expectations

- **Latency target: 7–9s measured**, with streaming so the first field appears in 2–3s.
  The original <5s criterion is withdrawn as unachievable for bilingual extraction on a
  T4 with a 3B model; it was set against arithmetic that did not hold.
- **`status` semantics unchanged.** Rule order, the two-pass agreement check, and the
  `missing`-outranks-`review` invariant all stand.
- **Occupation (المهنة)** is deliberately NOT extracted. It appears on resident cards
  only, is free text, and nothing in the demo needs it.
