//! Query distillation: turning a free-text generation brief into a short,
//! keyword-focused Arabic retrieval query via Haiku.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::anthropic::completion::CLAUDE_HAIKU_4_5;

use crate::ai_providers::anthropic_client;

/// System prompt for the query-distillation pass (Haiku). The user's brief is a
/// *generation* instruction — it carries imperative verbs («اشرح»), book/edition
/// titles, author names, and formatting asks. Embedding it verbatim drags the
/// vector toward front-matter and title pages that repeat those proper nouns,
/// not toward the chunks that actually answer the topic. This pass strips the
/// brief down to the bare Arabic topical terms BGE-M3 should match against.
const QUERY_DISTILL_PROMPT: &str = "\
أنت محرّك استخلاص استعلامات بحث دلالي على مكتبة كتب عربية.
حوّل وصف المستخدم إلى استعلام بحث عربي قصير ومركّز يلتقط الموضوع الفقهي/العلمي الجوهري.

القواعد:
- أخرج المصطلحات والعبارات المفتاحية الموضوعية فقط (المسائل، الأبواب، الأحكام).
- احذف عناوين الكتب وأسماء المؤلفين والطبعات والمحققين.
- احذف أفعال الطلب وكلمات التنسيق (مثل: اشرح، بيّن، اذكر، مع بيان، صفة، أعدّ عرضًا).
- بدون تشكيل، وبدون علامات ترقيم، وبدون أقواس، وبدون شروح.
- أخرج الاستعلام في سطر واحد فقط، دون أي نص إضافي.";

/// Distill a free-text brief into a short, keyword-focused Arabic retrieval
/// query using Haiku. Falls back to the original brief if the model returns an
/// empty line, so retrieval never runs on nothing.
pub(crate) async fn distill_query(brief: &str) -> anyhow::Result<String> {
    let agent = anthropic_client()
        .agent(CLAUDE_HAIKU_4_5)
        .preamble(QUERY_DISTILL_PROMPT)
        .max_tokens(256)
        .build();
    let distilled = agent.prompt(brief).await?;
    let distilled = distilled.trim();
    if distilled.is_empty() {
        Ok(brief.to_string())
    } else {
        Ok(distilled.to_string())
    }
}
