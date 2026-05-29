base_path <- "./data/new_data"
figure_dir <- file.path(base_path, "figures")

face_mask_output <- file.path(figure_dir, "face_mask_top_predictors_no_state.png")
protective_output <- file.path(figure_dir, "protective_behaviour_top_predictors_no_state.png")

feature_importance_sources <- list(
  face_mask = list(
    list(label = "All-time", path = file.path(base_path, "result_xgb", "model_1_all_time_face_mask", "feature_importances.csv")),
    list(label = "Non-mandate", path = file.path(base_path, "result_xgb", "model_1a_non_mandate_face_mask", "feature_importances.csv")),
    list(label = "Mandate", path = file.path(base_path, "result_xgb", "model_1b_mandate_face_mask", "feature_importances.csv"))
  ),
  protective = list(
    list(label = "All-time", path = file.path(base_path, "result_xgb", "model_2_all_time_protective_behaviour", "feature_importances.csv")),
    list(label = "Non-mandate", path = file.path(base_path, "result_xgb", "model_2a_non_mandate_protective_behaviour", "feature_importances.csv")),
    list(label = "Mandate", path = file.path(base_path, "result_xgb", "model_2b_mandate_protective_behaviour", "feature_importances.csv"))
  )
)

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

load_top_features_no_state <- function(source_list, top_n = 10) {
  frames <- list()

  for (source in source_list) {
    if (!file.exists(source$path)) {
      message("Skipped missing file: ", source$path)
      next
    }

    df <- read.csv(source$path, stringsAsFactors = FALSE)
    df <- df[!grepl("^state_", df$feature), ]
    df <- df[order(-df$importance), ]
    df <- head(df, top_n)
    df$period <- source$label
    frames[[length(frames) + 1]] <- df
  }

  if (length(frames) == 0) {
    stop("No feature importance files were found.")
  }

  frames
}

plot_feature_importance_panels_no_state <- function(source_key, output_path, title_text) {
  frames <- load_top_features_no_state(feature_importance_sources[[source_key]])
  panel_count <- length(frames)
  panel_colors <- c("#4E79A7", "#F28E2B", "#59A14F")

  png(output_path, width = 2100, height = 900, res = 180)
  par(mfrow = c(1, panel_count), mar = c(5, 10, 4, 2) + 0.1, oma = c(0, 0, 2, 0))

  for (i in seq_along(frames)) {
    df <- frames[[i]]
    df <- df[order(df$importance), ]

    barplot(
      df$importance,
      names.arg = df$feature,
      horiz = TRUE,
      las = 1,
      col = panel_colors[i],
      xlab = "Feature Importance",
      main = unique(df$period),
      cex.names = 0.8
    )
  }

  mtext(title_text, outer = TRUE, cex = 1.4, font = 2)
  dev.off()
}

main <- function() {
  plot_feature_importance_panels_no_state(
    source_key = "face_mask",
    output_path = face_mask_output,
    title_text = "Top XGBoost Predictors of Face Mask Behaviour (Excluding State Variables)"
  )

  plot_feature_importance_panels_no_state(
    source_key = "protective",
    output_path = protective_output,
    title_text = "Top XGBoost Predictors of Protective Behaviour (Excluding State Variables)"
  )

  message("Saved figures:")
  message(face_mask_output)
  message(protective_output)
}

main()
