import { useState } from "react";
import iconSuggested from "../assets/figma/icon-suggested.svg";
import iconPension from "../assets/figma/icon-pension.svg";
import iconEducation from "../assets/figma/icon-education.svg";
import iconHealth from "../assets/figma/icon-health.svg";
import iconJobs from "../assets/figma/icon-jobs.svg";

const TABS = ["Suggested", "Pension", "Health", "Education", "Jobs", "Family"];

// One curated prompt list per tab. "Suggested" is the original one-per-category
// mix from the design; the rest are the citizen's own per-category questions.
// No dedicated "Family" icon exists in the source design, so its rows reuse
// the pension icon (the closest existing glyph - elderly/household care).
const PROMPTS_BY_TAB = {
  Suggested: [
    { icon: iconPension, text: "My mother lives alone — is there a pension for her?" },
    { icon: iconEducation, text: "Free coaching for my daughter's exams" },
    { icon: iconHealth, text: "Cashless hospital treatment for my family" },
    { icon: iconJobs, text: "Government job after Class 12" },
  ],
  Pension: [
    { icon: iconPension, text: "My mother lives alone — is there a pension for her?" },
    { icon: iconPension, text: "Is there an old-age pension for my father?" },
    { icon: iconPension, text: "I am a widow — which pension can I apply for?" },
    { icon: iconPension, text: "How much is the monthly pension for senior citizens?" },
  ],
  Health: [
    { icon: iconHealth, text: "Cashless hospital treatment for my family" },
    { icon: iconHealth, text: "Does Ayushman Arogya cover my father's surgery?" },
    { icon: iconHealth, text: "How do I get a health card for my family?" },
    { icon: iconHealth, text: "Are medicines free at government hospitals?" },
  ],
  Education: [
    { icon: iconEducation, text: "Free coaching for my daughter's exams" },
    { icon: iconEducation, text: "Is there a scholarship for my son's college fees?" },
    { icon: iconEducation, text: "Can an orphaned child get free education support?" },
    { icon: iconEducation, text: "How do I apply for free UPSC or RAS coaching?" },
  ],
  Jobs: [
    { icon: iconJobs, text: "Government job after Class 12" },
    { icon: iconJobs, text: "Are there any current government job vacancies?" },
    { icon: iconJobs, text: "How do I apply for a junior assistant post?" },
    { icon: iconJobs, text: "What is the age limit for government jobs?" },
  ],
  Family: [
    { icon: iconPension, text: "My family has no ration card — how do we apply?" },
    { icon: iconPension, text: "I care for my elderly parents at home — any support scheme?" },
    { icon: iconPension, text: "Widowed with young children — what schemes can help my family?" },
    { icon: iconPension, text: "My family has 3 school-going children — what benefits apply?" },
  ],
};

export default function SuggestionsPanel({ onSend, disabled = false }) {
  const [activeTab, setActiveTab] = useState("Suggested");
  const prompts = PROMPTS_BY_TAB[activeTab];

  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 0, gap: 4 }}>
      <div
        style={{
          display: "flex",
          gap: 8,
          paddingBottom: 8,
          paddingLeft: 2,
          paddingRight: 2,
          borderBottom: "1px solid var(--color-tab-border)",
          overflowX: "auto",
        }}
      >
        {TABS.map((tab) => {
          const active = tab === activeTab;
          return (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                minHeight: 42,
                padding: "0 15px",
                borderRadius: 999,
                border: "none",
                background: active ? "var(--color-tab-active-bg)" : "transparent",
                cursor: "pointer",
                flexShrink: 0,
              }}
            >
              {tab === "Suggested" && <img src={iconSuggested} alt="" width={15} height={15} />}
              <span
                style={{
                  fontFamily: "var(--font-latin)",
                  fontWeight: active ? 600 : 500,
                  fontSize: 14.4,
                  color: active ? "var(--color-ink)" : "var(--color-ink-faint)",
                  whiteSpace: "nowrap",
                }}
              >
                {tab}
              </span>
            </button>
          );
        })}
      </div>

      <div style={{ display: "flex", flexDirection: "column" }}>
        {prompts.map((prompt, i) => (
          <button
            key={prompt.text}
            onClick={() => onSend(prompt.text)}
            disabled={disabled}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 15,
              minHeight: 60,
              padding: 16,
              border: "none",
              borderBottom: i < prompts.length - 1 ? "1px solid var(--color-tab-border)" : "none",
              background: "transparent",
              cursor: disabled ? "default" : "pointer",
              opacity: disabled ? 0.5 : 1,
              textAlign: "left",
              width: "100%",
            }}
          >
            <img src={prompt.icon} alt="" width={19} height={19} style={{ flexShrink: 0 }} />
            <span
              style={{
                fontFamily: "var(--font-latin)",
                fontWeight: 400,
                fontSize: 16.16,
                lineHeight: "22.624px",
                color: "var(--color-ink)",
              }}
            >
              {prompt.text}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
