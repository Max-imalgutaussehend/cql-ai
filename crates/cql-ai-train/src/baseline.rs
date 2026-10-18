//! Baselines the model must beat: a rule-based "anti-pattern" heuristic over the statement text only.

use crate::dataset::Example;

/// Predicts labels from keywords alone, the way a static anti-pattern linter would (no data-shape knowledge).
pub fn rules(e: &Example) -> [bool; cql_ai_model::N_LABELS] {
    let s = e.statement.to_ascii_uppercase();
    let is_select = s.starts_with("SELECT");
    let has_limit = s.contains(" LIMIT ");
    let mut l = [false; cql_ai_model::N_LABELS];
    l[0] = is_select && !has_limit && !s.contains("COUNT(");
    l[1] = s.contains("ALLOW FILTERING") || s.contains("COUNT(");
    l[2] = s.contains(" IN (")
        || s.contains(" IN ?")
        || s.contains("TOKEN(")
        || (is_select && !s.contains(" WHERE "));
    l[3] = false;
    l[4] = !is_select && (s.contains("USING TTL") || e.ctx[10] > 0.0);
    l[5] = s.contains("IF NOT EXISTS") || s.contains(" IF ");
    l
}
