"""Generating the values printed on a card.

Every value here is fabricated. The point of the corpus is coverage of the
SHAPES the model must learn to read - name length, letter combinations,
date formats - not resemblance to any real person or record.

The name generator is the part that matters. full_name_ar is the only
open-vocabulary field on the card and the only one the model has to
genuinely read rather than recognise, so this deliberately oversamples long
names: a seven-component name is where truncation showed up on real cards,
and a uniform sample would under-represent exactly that case.
"""

import random
from dataclasses import dataclass

GIVEN_MALE = [
    "زياد", "محمد", "أحمد", "علي", "سالم", "خالد", "سعيد", "ناصر", "يوسف",
    "إبراهيم", "حمد", "طلال", "ماجد", "فهد", "سلطان", "بدر", "راشد", "مازن",
    "قيس", "طارق", "عمر", "حسن", "حسين", "منصور", "هلال", "سيف", "بدرية",
    "عامر", "جابر", "مالك", "نبيل", "وليد", "أنور", "سامي", "رامي", "أيمن",
]

GIVEN_FEMALE = [
    "عهود", "فاطمة", "مريم", "عائشة", "نورة", "سارة", "هدى", "منى", "ريم",
    "شيخة", "بثينة", "أسماء", "لطيفة", "زينب", "خديجة", "سميرة", "نادية",
    "أمل", "رقية", "بدور", "جميلة", "سلمى", "ليلى", "وفاء", "هناء",
]

# "عبد" compounds count as one component but are printed as two words, which
# is a shape the model has to get right - it is where a name most often gets
# split or truncated mid-way.
ABD_COMPOUNDS = [
    "عبد الله", "عبد الرحمن", "عبد العزيز", "عبد الحي", "عبد الكريم",
    "عبد الرحيم", "عبد الملك", "عبد الوهاب", "عبد الحميد", "عبد الغفور",
]

FAMILY_OMANI = [
    "البلوشي", "الحارثي", "الشامسي", "الكندي", "المعمري", "الرواحي",
    "البوسعيدي", "الهنائي", "الزدجالي", "السيابي", "الغافري", "الريامي",
    "البطاشي", "النعماني", "الفارسي", "اللواتي", "الخروصي", "المقبالي",
    "الحجري", "الصوافي", "العبري", "الشكيلي", "الجابري", "الحبسي",
    "المحروقي", "الشحي", "الوهيبي", "الحوسني", "الرئيسي", "البادي",
]

# Resident cards carry non-Omani names, and their shape differs - fewer
# tribal "ال" prefixes, more multi-word family names.
FAMILY_EXPAT = [
    "حموده", "شعبان", "مرسي", "الشناوي", "عبد المقصود", "أبو الوفا",
    "الشربيني", "زغلول", "الدسوقي", "بركات", "خميس", "سليمان", "عوض",
    "جاد", "فرج", "منصور", "الحلو", "قنديل", "العطار", "الصياد",
]

PLACES_OMANI = [
    "مسقط", "ظفار", "مسندم", "البريمي", "الداخلية", "شمال الباطنة",
    "جنوب الباطنة", "شمال الشرقية", "جنوب الشرقية", "الظاهرة", "الوسطى",
    "صلالة", "نزوى", "صحار", "صور", "عبري", "الرستاق", "إبراء",
]

PLACES_EXPAT = [
    "جمهورية مصر العربية", "الهند", "باكستان", "بنغلاديش", "الفلبين",
    "سريلانكا", "السودان", "الأردن", "سوريا", "اليمن", "نيبال",
    "المملكة المغربية", "الجمهورية التونسية", "لبنان", "العراق",
]

OCCUPATIONS = [
    "إلتحاق بالأقارب", "مهندس", "محاسب", "فني", "سائق", "طبيب", "مدرس",
    "عامل", "مشرف", "بائع", "حارس أمن", "طباخ", "كهربائي", "نجار",
]


@dataclass(frozen=True)
class CardContent:
    """The six extracted fields, plus the two we print but never extract."""

    card_type: str
    full_name_ar: str
    id_number: str
    date_of_birth: str
    expiry_date: str
    place_of_birth_ar: str
    occupation_ar: str | None
    serial: str

    def ground_truth(self) -> dict:
        """Exactly the production JSON. Nothing else belongs in a label."""
        return {
            "card_type": self.card_type,
            "full_name_ar": self.full_name_ar,
            "id_number": self.id_number,
            "date_of_birth": self.date_of_birth,
            "expiry_date": self.expiry_date,
            "place_of_birth_ar": self.place_of_birth_ar,
        }


# Component counts weighted towards the long end. Real Omani names run five
# to eight components, and the failures we have observed were all at that
# end, so a uniform 3-9 sample would spend most of its data where the model
# already succeeds.
_COMPONENT_WEIGHTS = {3: 4, 4: 10, 5: 20, 6: 24, 7: 22, 8: 14, 9: 6}


def _name_components(rng: random.Random, citizen: bool) -> list[str]:
    count = rng.choices(
        list(_COMPONENT_WEIGHTS), weights=list(_COMPONENT_WEIGHTS.values())
    )[0]
    female = rng.random() < 0.45
    given = GIVEN_FEMALE if female else GIVEN_MALE
    family = FAMILY_OMANI if citizen else FAMILY_EXPAT

    parts = [rng.choice(given)]
    # Middle components are the father/grandfather chain: male given names,
    # with the عبد compounds appearing at a realistic rate.
    while len(parts) < count - 1:
        choice = (
            rng.choice(ABD_COMPOUNDS) if rng.random() < 0.22 else rng.choice(GIVEN_MALE)
        )
        # Adjacent repeats do occur in real names but read as a generation
        # artefact, and a model should not learn that doubling is common.
        if choice != parts[-1]:
            parts.append(choice)
    parts.append(rng.choice(family))
    return parts


def _date(rng: random.Random, start_year: int, end_year: int) -> tuple[str, str]:
    """Return (iso, printed). The card prints DD/MM/YYYY; ground truth is ISO."""
    year = rng.randint(start_year, end_year)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)  # avoids month-length special cases entirely
    return f"{year:04d}-{month:02d}-{day:02d}", f"{day:02d}/{month:02d}/{year:04d}"


def generate(rng: random.Random) -> tuple[CardContent, dict]:
    """Return the card content and the printed (DD/MM/YYYY) forms of its dates."""
    citizen = rng.random() < 0.5
    card_type = "citizen" if citizen else "resident"

    dob_iso, dob_printed = _date(rng, 1945, 2007)
    # Expiry follows the ISSUE date, not the date of birth. Deriving it from
    # the holder's age produced cards issued to a 2007 birth that expired in
    # 2076 - a validator would pass it, since expiry does follow birth, and
    # the model would learn a distribution no card has. Omani cards run five
    # to ten years, and a corpus contains recently expired ones too.
    exp_iso, exp_printed = _date(rng, 2022, 2035)

    content = CardContent(
        card_type=card_type,
        full_name_ar=" ".join(_name_components(rng, citizen)),
        id_number=f"{rng.randint(10_000_000, 99_999_999)}",
        date_of_birth=dob_iso,
        expiry_date=exp_iso,
        place_of_birth_ar=rng.choice(PLACES_OMANI if citizen else PLACES_EXPAT),
        # Printed on resident cards only. Never extracted, but it must be on
        # the card: it is the row directly below the name, and it is what
        # bounds the name when anything reads the card line by line.
        occupation_ar=None if citizen else rng.choice(OCCUPATIONS),
        serial="".join(rng.choice("0123456789ABCDEF") for _ in range(16)),
    )
    return content, {"date_of_birth": dob_printed, "expiry_date": exp_printed}
