use chrono::{Datelike, NaiveDate};

pub fn daily_note_day_tag(date: NaiveDate) -> String {
    format!("day:{date}")
}

pub fn daily_note_weekday_tag(date: NaiveDate) -> String {
    format!(
        "weekday:{}",
        date.format("%a").to_string().to_ascii_lowercase()
    )
}

pub fn render_daily_note_template(template: &str, date: NaiveDate) -> String {
    let yesterday = date.pred_opt().unwrap_or(date);
    let tomorrow = date.succ_opt().unwrap_or(date);

    template
        .replace("{{date}}", &date.to_string())
        .replace("{{weekday}}", &date.format("%A").to_string())
        .replace("{{month}}", &date.format("%B").to_string())
        .replace("{{year}}", &date.year().to_string())
        .replace("{{iso_week}}", &date.iso_week().week().to_string())
        .replace("{{yesterday}}", &yesterday.to_string())
        .replace("{{tomorrow}}", &tomorrow.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_template_tokens() {
        let date = NaiveDate::from_ymd_opt(2026, 2, 6).expect("valid date");
        let rendered = render_daily_note_template(
            "Note {{date}} {{weekday}} {{month}} {{year}} {{iso_week}} {{yesterday}} {{tomorrow}}",
            date,
        );

        assert!(rendered.contains("2026-02-06"));
        assert!(rendered.contains("Friday"));
        assert!(rendered.contains("February"));
        assert!(rendered.contains("2026"));
        assert!(rendered.contains("6"));
        assert!(rendered.contains("2026-02-05"));
        assert!(rendered.contains("2026-02-07"));
    }

    #[test]
    fn builds_stable_tags() {
        let date = NaiveDate::from_ymd_opt(2026, 2, 6).expect("valid date");
        assert_eq!(daily_note_day_tag(date), "day:2026-02-06");
        assert_eq!(daily_note_weekday_tag(date), "weekday:fri");
    }
}
