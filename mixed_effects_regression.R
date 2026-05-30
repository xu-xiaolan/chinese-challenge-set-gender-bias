# ============================================================
# Mixed-Effects Logistic Regression
# Gender Bias in Chinese-Portuguese Automated Translation
# EAMT/GITT 2026
#
# Requirements:
#   install.packages("lme4")
#
# Usage:
#   Rscript mixed_effects_regression.R
# ============================================================

library(lme4)

# ── 1. Load data ─────────────────────────────────────────────
df <- read.csv("coreference_long.csv")

# ── 2. Set reference categories ──────────────────────────────
# Reference: masculine cue, simple sentence, anaphoric,
#            NMT system, gender-neutral stereotype
df$target_gender <- relevel(factor(df$target_gender), ref = "m")
df$complexity    <- relevel(factor(df$complexity),    ref = "smp")
df$position      <- relevel(factor(df$position),      ref = "ante")
df$system_type   <- relevel(factor(df$system_type),   ref = "NMT")
df$bias_label    <- relevel(factor(df$bias_label),    ref = "N")

cat("N =", nrow(df), "\n")
cat("Occupations (random effect):", length(unique(df$occupation)), "\n\n")

# ── 3. Fit mixed-effects logistic regression ──────────────────
# Fixed effects: target_gender, complexity, position,
#                system_type, bias_label
# Random effect: occupation (random intercept)
model <- glmer(
  correct ~ target_gender + complexity + position +
            system_type + bias_label + (1 | occupation),
  data    = df,
  family  = binomial,
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl   = list(maxfun = 2e5)
  )
)

# ── 4. Print summary ──────────────────────────────────────────
print(summary(model))

# ── 5. Extract odds ratios and confidence intervals ───────────
coefs <- fixef(model)
ci    <- confint(model, parm = "beta_", method = "Wald")
or    <- exp(coefs)
or_lo <- exp(ci[, 1])
or_hi <- exp(ci[, 2])

# p-values from summary
p_vals <- summary(model)$coefficients[, 4]

sig_stars <- function(p) {
  if (p < .001) return("***")
  if (p < .01)  return("**")
  if (p < .05)  return("*")
  if (p < .10)  return(".")
  return("")
}

cat("\n============================================================\n")
cat("MIXED-EFFECTS LOGISTIC REGRESSION — ODDS RATIOS\n")
cat("Random effect: occupation (intercept)\n")
cat("============================================================\n")
cat(sprintf("%-35s %7s %6s %14s %9s %4s\n",
            "Predictor", "beta", "OR", "95% CI", "p", ""))
cat(paste(rep("-", 80), collapse=""), "\n")

for (nm in names(coefs)) {
  if (nm == "(Intercept)") next
  cat(sprintf("%-35s %+7.3f %6.2f [%5.2f, %5.2f] %9.4f %s\n",
              nm,
              coefs[nm],
              or[nm],
              or_lo[nm],
              or_hi[nm],
              p_vals[nm],
              sig_stars(p_vals[nm])))
}

cat(paste(rep("-", 80), collapse=""), "\n")
cat(sprintf("N = %d\n", nrow(df)))
cat(sprintf("AIC = %.1f\n", AIC(model)))
cat(sprintf("Random effect variance (occupation) = %.4f\n",
            as.data.frame(VarCorr(model))$vcov[1]))
cat("*** p<.001  ** p<.01  * p<.05  . p<.10\n")

# ── 6. Supplementary: gender x complexity interaction test ────
model_int <- glmer(
  correct ~ target_gender * complexity + position +
            system_type + bias_label + (1 | occupation),
  data    = df,
  family  = binomial,
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl   = list(maxfun = 2e5)
  )
)

lr_test <- anova(model, model_int)
cat(sprintf(
  "\nInteraction test (gender x complexity):\n  chi2 = %.3f, df = %d, p = %.4f\n",
  lr_test$Chisq[2],
  lr_test$Df[2],
  lr_test$`Pr(>Chisq)`[2]
))
