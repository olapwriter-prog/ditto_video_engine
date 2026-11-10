//! Writing systems: which script a piece of text is in, and which languages use it.
//!
//! A sign board in India usually carries two or three scripts at once: the state's
//! language, Hindi and English. Anything that reads text (signs, receipts, screenshots)
//! has to split such text into runs of one script before it can pick a recogniser, a
//! voice or a translation. This module is that split, plus digits: every Indian script
//! has its own, and a speed limit or an amount must come out as the same number.
//!
//! Coverage: the scripts of India's 22 scheduled languages and English, and the Greek and
//! Cyrillic scripts used in Europe besides Latin. Script detection is exact (it is a
//! property of the characters). The language is **not** detected: several languages share
//! a script, so a run reports the candidates and nothing more, apart from Assamese, whose
//! own letters tell it from Bengali.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Devanagari,
    /// Bengali and Assamese share it.
    Bengali,
    Gurmukhi,
    Gujarati,
    Odia,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    /// Urdu, Kashmiri and Sindhi in India.
    Arabic,
    /// Manipuri (Meitei).
    MeiteiMayek,
    /// Santali.
    OlChiki,
    Greek,
    Cyrillic,
}

/// A language and the scripts it is written in, the usual one first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Language {
    /// ISO 639 code.
    pub code: &'static str,
    pub name: &'static str,
    pub scripts: &'static [Script],
}

use Script::*;

/// India's 22 scheduled languages, and English.
pub const INDIA: &[Language] = &[
    Language {
        code: "as",
        name: "Assamese",
        scripts: &[Bengali],
    },
    Language {
        code: "bn",
        name: "Bengali",
        scripts: &[Bengali],
    },
    Language {
        code: "brx",
        name: "Bodo",
        scripts: &[Devanagari],
    },
    Language {
        code: "doi",
        name: "Dogri",
        scripts: &[Devanagari],
    },
    Language {
        code: "gu",
        name: "Gujarati",
        scripts: &[Gujarati],
    },
    Language {
        code: "hi",
        name: "Hindi",
        scripts: &[Devanagari],
    },
    Language {
        code: "kn",
        name: "Kannada",
        scripts: &[Kannada],
    },
    Language {
        code: "ks",
        name: "Kashmiri",
        scripts: &[Arabic, Devanagari],
    },
    Language {
        code: "kok",
        name: "Konkani",
        scripts: &[Devanagari],
    },
    Language {
        code: "mai",
        name: "Maithili",
        scripts: &[Devanagari],
    },
    Language {
        code: "ml",
        name: "Malayalam",
        scripts: &[Malayalam],
    },
    Language {
        code: "mni",
        name: "Manipuri",
        scripts: &[MeiteiMayek, Bengali],
    },
    Language {
        code: "mr",
        name: "Marathi",
        scripts: &[Devanagari],
    },
    Language {
        code: "ne",
        name: "Nepali",
        scripts: &[Devanagari],
    },
    Language {
        code: "or",
        name: "Odia",
        scripts: &[Odia],
    },
    Language {
        code: "pa",
        name: "Punjabi",
        scripts: &[Gurmukhi],
    },
    Language {
        code: "sa",
        name: "Sanskrit",
        scripts: &[Devanagari],
    },
    Language {
        code: "sat",
        name: "Santali",
        scripts: &[OlChiki, Devanagari],
    },
    Language {
        code: "sd",
        name: "Sindhi",
        scripts: &[Arabic, Devanagari],
    },
    Language {
        code: "ta",
        name: "Tamil",
        scripts: &[Tamil],
    },
    Language {
        code: "te",
        name: "Telugu",
        scripts: &[Telugu],
    },
    Language {
        code: "ur",
        name: "Urdu",
        scripts: &[Arabic],
    },
    Language {
        code: "en",
        name: "English",
        scripts: &[Latin],
    },
];

impl Script {
    pub const ALL: &'static [Script] = &[
        Latin,
        Devanagari,
        Bengali,
        Gurmukhi,
        Gujarati,
        Odia,
        Tamil,
        Telugu,
        Kannada,
        Malayalam,
        Arabic,
        MeiteiMayek,
        OlChiki,
        Greek,
        Cyrillic,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Latin => "Latin",
            Devanagari => "Devanagari",
            Bengali => "Bengali-Assamese",
            Gurmukhi => "Gurmukhi",
            Gujarati => "Gujarati",
            Odia => "Odia",
            Tamil => "Tamil",
            Telugu => "Telugu",
            Kannada => "Kannada",
            Malayalam => "Malayalam",
            Arabic => "Perso-Arabic",
            MeiteiMayek => "Meitei Mayek",
            OlChiki => "Ol Chiki",
            Greek => "Greek",
            Cyrillic => "Cyrillic",
        }
    }

    /// The script a character belongs to. `None` for digits shared by all scripts,
    /// spaces, punctuation and anything outside the covered scripts.
    pub fn of(c: char) -> Option<Script> {
        if digit_value(c).is_some() {
            return None; // a number is not in a language
        }
        Some(match c as u32 {
            0x41..=0x5A
            | 0x61..=0x7A
            | 0xAA
            | 0xBA
            | 0xC0..=0xD6
            | 0xD8..=0xF6
            | 0xF8..=0x24F
            | 0x1E00..=0x1EFF => Latin,
            0x0370..=0x0373
            | 0x0376..=0x0377
            | 0x037B..=0x037D
            | 0x037F
            | 0x0386
            | 0x0388..=0x03FF
            | 0x1F00..=0x1FFF => Greek,
            0x0400..=0x052F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => Cyrillic,
            // U+0964 and U+0965, the danda marks, are punctuation shared by many scripts
            0x0900..=0x0963 | 0x0966..=0x097F | 0xA8E0..=0xA8FF | 0x1CD0..=0x1CFF => Devanagari,
            0x0980..=0x09FF => Bengali,
            0x0A00..=0x0A7F => Gurmukhi,
            0x0A80..=0x0AFF => Gujarati,
            0x0B00..=0x0B7F => Odia,
            0x0B80..=0x0BFF => Tamil,
            0x0C00..=0x0C7F => Telugu,
            0x0C80..=0x0CFF => Kannada,
            0x0D00..=0x0D7F => Malayalam,
            0x0600..=0x06FF
            | 0x0750..=0x077F
            | 0x08A0..=0x08FF
            | 0xFB50..=0xFDFF
            | 0xFE70..=0xFEFF => Arabic,
            0xABC0..=0xABFF | 0xAAE0..=0xAAFF => MeiteiMayek,
            0x1C50..=0x1C7F => OlChiki,
            _ => return None,
        })
    }

    /// Languages of [`INDIA`] written in this script, those that mainly use it first.
    pub fn languages(self) -> Vec<&'static Language> {
        let mut v: Vec<&Language> = INDIA.iter().filter(|l| l.scripts.contains(&self)).collect();
        v.sort_by_key(|l| l.scripts[0] != self);
        v
    }

    /// Written right to left.
    pub fn right_to_left(self) -> bool {
        self == Arabic
    }
}

/// The value of a decimal digit in any covered script, ASCII included.
pub fn digit_value(c: char) -> Option<u32> {
    const ZEROS: &[u32] = &[
        0x30,   // ASCII
        0x0966, // Devanagari
        0x09E6, // Bengali
        0x0A66, // Gurmukhi
        0x0AE6, // Gujarati
        0x0B66, // Odia
        0x0BE6, // Tamil
        0x0C66, // Telugu
        0x0CE6, // Kannada
        0x0D66, // Malayalam
        0x0660, // Arabic-Indic
        0x06F0, // Eastern Arabic-Indic (Urdu, Kashmiri, Sindhi)
        0xABF0, // Meitei Mayek
        0x1C50, // Ol Chiki
    ];
    let u = c as u32;
    ZEROS
        .iter()
        .find(|z| (**z..**z + 10).contains(&u))
        .map(|z| u - z)
}

/// The text with every script's digits written as `0` to `9`. Nothing else changes.
pub fn ascii_digits(text: &str) -> String {
    text.chars()
        .map(|c| {
            digit_value(c)
                .and_then(|d| char::from_digit(d, 10))
                .unwrap_or(c)
        })
        .collect()
}

/// A stretch of text in one script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    pub script: Script,
    pub text: String,
}

impl Run {
    /// Languages this run can be in. Assamese text that uses Assamese's own letters
    /// (ৰ, ৱ) is reported as Assamese first.
    pub fn languages(&self) -> Vec<&'static Language> {
        let mut v = self.script.languages();
        if self.script == Bengali
            && self
                .text
                .chars()
                .any(|c| matches!(c, '\u{09F0}' | '\u{09F1}'))
        {
            v.sort_by_key(|l| l.code != "as");
        }
        v
    }
}

/// Split text into runs of one script. Digits, spaces and punctuation stay with the run
/// they are in; at a change of script they go to the run before, and leading ones to the
/// first run. Text with no letters of a covered script gives no runs.
pub fn runs(text: &str) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    let mut pending = String::new(); // neutral characters seen before any letter
    for c in text.chars() {
        match (Script::of(c), out.last_mut()) {
            (Some(s), Some(last)) if last.script == s => last.text.push(c),
            (Some(s), _) => {
                let mut t = std::mem::take(&mut pending);
                t.push(c);
                out.push(Run { script: s, text: t });
            }
            (None, Some(last)) => last.text.push(c),
            (None, None) => pending.push(c),
        }
    }
    for r in &mut out {
        r.text = r.text.trim().to_string();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each scheduled language's own name for itself, in its usual script.
    const NAMES: &[(&str, &str, Script)] = &[
        ("as", "অসমীয়া", Bengali),
        ("bn", "বাংলা", Bengali),
        ("brx", "बड़ो", Devanagari),
        ("doi", "डोगरी", Devanagari),
        ("gu", "ગુજરાતી", Gujarati),
        ("hi", "हिन्दी", Devanagari),
        ("kn", "ಕನ್ನಡ", Kannada),
        ("ks", "کٲشُر", Arabic),
        ("kok", "कोंकणी", Devanagari),
        ("mai", "मैथिली", Devanagari),
        ("ml", "മലയാളം", Malayalam),
        ("mni", "ꯃꯩꯇꯩꯂꯣꯟ", MeiteiMayek),
        ("mr", "मराठी", Devanagari),
        ("ne", "नेपाली", Devanagari),
        ("or", "ଓଡ଼ିଆ", Odia),
        ("pa", "ਪੰਜਾਬੀ", Gurmukhi),
        ("sa", "संस्कृतम्", Devanagari),
        ("sat", "ᱥᱟᱱᱛᱟᱲᱤ", OlChiki),
        ("sd", "سنڌي", Arabic),
        ("ta", "தமிழ்", Tamil),
        ("te", "తెలుగు", Telugu),
        ("ur", "اردو", Arabic),
        ("en", "English", Latin),
    ];

    #[test]
    fn all_22_scheduled_languages_and_english_are_covered() {
        assert_eq!(INDIA.len(), 23);
        assert_eq!(NAMES.len(), 23);
        let mut codes: Vec<&str> = INDIA.iter().map(|l| l.code).collect();
        codes.sort();
        codes.dedup();
        assert_eq!(codes.len(), 23, "a language is listed twice");
        for (code, name, script) in NAMES {
            let lang = INDIA
                .iter()
                .find(|l| l.code == *code)
                .unwrap_or_else(|| panic!("{code} missing"));
            assert_eq!(lang.scripts[0], *script, "{code}");
            // the name is one run, wholly in that script, and the language is a candidate
            let r = runs(name);
            assert_eq!(r.len(), 1, "{code}: {r:?}");
            assert_eq!(
                (r[0].script, r[0].text.as_str()),
                (*script, *name),
                "{code}"
            );
            assert!(r[0].languages().iter().any(|l| l.code == *code), "{code}");
        }
        // every script of every language is one this module can detect
        for l in INDIA {
            assert!(l.scripts.iter().all(|s| Script::ALL.contains(s)));
        }
    }

    #[test]
    fn a_sign_board_in_three_scripts_is_split() {
        // a station name as boards show it: state language, Hindi, English
        let r = runs("హైదరాబాద్ हैदराबाद HYDERABAD");
        assert_eq!(
            r.iter()
                .map(|x| (x.script, x.text.as_str()))
                .collect::<Vec<_>>(),
            [
                (Telugu, "హైదరాబాద్"),
                (Devanagari, "हैदराबाद"),
                (Latin, "HYDERABAD")
            ]
        );
        let r = runs("சென்னை 12 கி.மீ / Chennai 12 km");
        assert_eq!(r.len(), 2);
        assert_eq!(
            (r[0].script, r[0].text.as_str()),
            (Tamil, "சென்னை 12 கி.மீ /")
        );
        assert_eq!((r[1].script, r[1].text.as_str()), (Latin, "Chennai 12 km"));
        // numbers and punctuation alone are not a script
        assert!(runs("40 - 60 !").is_empty());
        assert!(runs("").is_empty());
        // leading neutral characters go to the first run
        assert_eq!(runs("₹ 120 मात्र")[0].text, "₹ 120 मात्र");
    }

    #[test]
    fn candidates_are_ordered_and_assamese_is_told_from_bengali() {
        assert_eq!(Devanagari.languages()[0].scripts[0], Devanagari);
        let codes = |s: Script| s.languages().iter().map(|l| l.code).collect::<Vec<_>>();
        assert_eq!(codes(Telugu), ["te"]);
        assert_eq!(codes(Arabic), ["ks", "sd", "ur"]);
        assert_eq!(codes(Bengali), ["as", "bn", "mni"]); // Manipuri's second script last
        assert_eq!(codes(Devanagari).len(), 11);
        assert!(codes(Greek).is_empty() && codes(Cyrillic).is_empty());
        // "ৰ" is used by Assamese, not by Bengali
        assert_eq!(runs("গুৱাহাটী ৰে'ল")[0].languages()[0].code, "as");
        assert_eq!(
            runs("কলকাতা")[0]
                .languages()
                .iter()
                .map(|l| l.code)
                .collect::<Vec<_>>(),
            ["as", "bn", "mni"]
        );
        assert!(Arabic.right_to_left() && !Telugu.right_to_left());
    }

    #[test]
    fn digits_of_every_script_are_the_same_numbers() {
        for (native, script) in [
            ("०१२३४५६७८९", "Devanagari"),
            ("০১২৩৪৫৬৭৮৯", "Bengali"),
            ("੦੧੨੩੪੫੬੭੮੯", "Gurmukhi"),
            ("૦૧૨૩૪૫૬૭૮૯", "Gujarati"),
            ("୦୧୨୩୪୫୬୭୮୯", "Odia"),
            ("௦௧௨௩௪௫௬௭௮௯", "Tamil"),
            ("౦౧౨౩౪౫౬౭౮౯", "Telugu"),
            ("೦೧೨೩೪೫೬೭೮೯", "Kannada"),
            ("൦൧൨൩൪൫൬൭൮൯", "Malayalam"),
            ("٠١٢٣٤٥٦٧٨٩", "Arabic-Indic"),
            ("۰۱۲۳۴۵۶۷۸۹", "Eastern Arabic-Indic"),
            ("꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹", "Meitei Mayek"),
            ("᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙", "Ol Chiki"),
            ("0123456789", "ASCII"),
        ] {
            assert_eq!(ascii_digits(native), "0123456789", "{script}");
        }
        // a speed limit and an amount, as written
        assert_eq!(ascii_digits("४० कि.मी./घं."), "40 कि.मी./घं.");
        assert_eq!(ascii_digits("₹ ౧,౨౫౦.౫౦"), "₹ 1,250.50");
        assert_eq!(digit_value('x'), None);
        assert_eq!(digit_value('\u{0965}'), None); // a danda, right after the Devanagari block's letters
                                                   // digits are numbers, not text in a script: a sign showing only "४०" has no language
        assert_eq!(Script::of('᱐'), None);
        assert_eq!(Script::of('४'), None);
        assert_eq!(Script::of('ᱚ'), Some(OlChiki));
        assert!(runs("४०").is_empty());
    }

    #[test]
    fn european_scripts() {
        let r = runs("Αθήνα Athens София Sofia Zürich Łódź");
        assert_eq!(
            r.iter().map(|x| x.script).collect::<Vec<_>>(),
            [Greek, Latin, Cyrillic, Latin]
        );
        assert_eq!(r[3].text, "Sofia Zürich Łódź");
        // the multiplication and division signs sit inside the Latin-1 letters
        assert_eq!(Script::of('×'), None);
        assert_eq!(Script::of('÷'), None);
    }

    #[test]
    fn every_character_has_at_most_one_script_and_names_are_distinct() {
        let mut names: Vec<&str> = Script::ALL.iter().map(|s| s.name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), Script::ALL.len());
        // every script is reachable from some character
        for s in Script::ALL {
            assert!(
                (0..0x1_0000u32)
                    .filter_map(char::from_u32)
                    .any(|c| Script::of(c) == Some(*s)),
                "{}",
                s.name()
            );
        }
    }
}
