//! Entity scoping: pulling a named author/source out of the brief and turning
//! it into a LanceDB filter that *biases* retrieval toward that entity.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::anthropic::completion::CLAUDE_HAIKU_4_5;

use crate::ai_providers::anthropic_client;

/// A named author/person (with known honorifics) and/or book/source extracted
/// from the brief, used to *bias* retrieval toward that entity. Empty everywhere
/// means "no entity named" — retrieval then behaves exactly as the topical path.
#[derive(Debug, Default)]
pub(crate) struct Scope {
    pub person: Option<String>,
    /// Honorifics / alternate names the person is cited by in fiqh texts
    /// (e.g. «شيخ الإسلام», «تقي الدين» for ابن تيمية). Lets us catch his
    /// *position* where it is quoted inside other authors' books, not just the
    /// books he himself wrote.
    pub aliases: Vec<String>,
    pub source: Option<String>,
}

impl Scope {
    fn is_empty(&self) -> bool {
        self.person.is_none() && self.source.is_none()
    }
}

/// Haiku prompt that pulls a named author + their honorifics + a named source
/// out of the brief. Deliberately the inverse of the query-distillation prompt,
/// which *strips* these — here we want them, to scope retrieval.
const SCOPE_EXTRACT_PROMPT: &str = "\
أنت مستخرج كيانات من طلب المستخدم.
حدّد إن كان الطلب يذكر مؤلفًا/عالِمًا معيّنًا، أو كتابًا/متنًا/مصدرًا معيّنًا.
أخرج ثلاثة أسطر فقط، حرفيًا بهذا الشكل، واستخدم --- لِما لم يُذكر:
person: <اسم المؤلف أو العالِم كما هو معروف، أو --->
aliases: <ألقابه وكُناه المشهورة التي يُذكر بها في كتب الفقه مفصولة بفواصل، أو --->
source: <اسم الكتاب أو المتن، أو --->
لا تكتب أي شيء آخر.";

/// Extract a [`Scope`] from the brief. Never fails the run: any error or an empty
/// answer yields an empty scope (no biasing — pure topical retrieval).
pub(crate) async fn extract_scope(brief: &str) -> Scope {
    let agent = anthropic_client()
        .agent(CLAUDE_HAIKU_4_5)
        .preamble(SCOPE_EXTRACT_PROMPT)
        .max_tokens(256)
        .build();
    let raw = match agent.prompt(brief).await {
        Ok(s) => s,
        Err(_) => return Scope::default(),
    };

    // Pull the value after a `label:` line; treat `---`/empty as unset.
    let field = |label: &str| -> Option<String> {
        raw.lines()
            .find_map(|l| l.trim().strip_prefix(label))
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty() && v != "---" && v != "--")
    };

    let aliases = field("aliases:")
        .map(|s| {
            s.split([',', '،'])
                .map(|a| a.trim().to_string())
                .filter(|a| !a.is_empty() && a != "---")
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Scope { person: field("person:"), aliases, source: field("source:") }
}

/// Build a LanceDB SQL filter matching the named entity, or `None` if none was
/// named. Person matching spans two senses: books he *authored*
/// (`main_author_name_ar`) and chunks that *cite him* by name or honorific
/// (`text LIKE`). Source matching is on `title_ar`. Clauses are OR'd so any match
/// keeps the row; the topical hybrid query still ANDs in relevance.
pub(crate) fn scope_filter(scope: &Scope) -> Option<String> {
    if scope.is_empty() {
        return None;
    }
    let esc = |s: &str| s.replace('\'', "''");
    let mut clauses: Vec<String> = Vec::new();

    if let Some(p) = &scope.person {
        clauses.push(format!("main_author_name_ar LIKE '%{}%'", esc(p)));
        let mut names: Vec<&str> = std::iter::once(p.as_str())
            .chain(scope.aliases.iter().map(String::as_str))
            .collect();
        names.sort();
        names.dedup();
        for n in names {
            clauses.push(format!("text LIKE '%{}%'", esc(n)));
        }
    }
    if let Some(s) = &scope.source {
        clauses.push(format!("title_ar LIKE '%{}%'", esc(s)));
    }

    Some(clauses.join(" OR "))
}
