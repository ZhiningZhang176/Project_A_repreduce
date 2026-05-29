base_path <- "./data/new_data"
figure_dir <- file.path(base_path, "figures")

comparison_output <- file.path(figure_dir, "model_performance_comparison.png")
face_mask_output <- file.path(figure_dir, "face_mask_top_predictors.png")
protective_output <- file.path(figure_dir, "protective_behaviour_top_predictors.png")

model_result_files <- list(
  "Logistic Regression" = file.path(base_path, "result_logistic", "all_model_results.csv"),
  "Decision Tree" = file.path(base_path, "result_tree", "all_model_results.csv"),
  "Random Forest" = file.path(base_path, "result_rf", "all_model_results.csv"),
  "XGBoost" = file.path(base_path, "result_xgb", "all_model_results.csv")
)

dataset_order <- c(
  "all_time_face_mask",
  "non_mandate_face_mask",
  "mandate_face_mask",
  "all_time_protective_behaviour",
  "non_mandate_protective_behaviour",
  "mandate_protective_behaviour"
)

dataset_labels <- c(
  "Face mask\nAll-time",
  "Face mask\nNon-mandate",
  "Face mask\nMandate",
  "Protective\nAll-time",
  "Protective\nNon-mandate",
  "Protective\nMandate"
)

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

load_model_results <- function() {
  frames <- list()

  for (model_family in names(model_result_files)) {
    file_path <- model_result_files[[model_family]]

    if (!file.exists(file_path)) {
      message("Skipped missing file: ", file_path)
      next
    }

    df <- read.csv(file_path, stringsAsFactors = FALSE)
    df$model_family <- model_family
    frames[[length(frames) + 1]] <- df
  }

  if (length(frames) == 0) {
    stop("No model result files were found.")
  }

  combined_df <- do.call(rbind, frames)
  combined_df <- combined_df[combined_df$dataset_scope %in% dataset_order, ]
  combined_df$dataset_scope <- factor(combined_df$dataset_scope, levels = dataset_order)
  combined_df
}

plot_model_performance <- function(combined_df) {
  available_models <- unique(combined_df$model_family)
  available_models <- available_models[order(match(available_models, names(model_result_files)))]

  plot_matrix <- matrix(
    NA_real_,
    nrow = length(available_models),
    ncol = length(dataset_order),
    dimnames = list(available_models, dataset_order)
  )

  for (i in seq_len(nrow(combined_df))) {
    row <- combined_df[i, ]
    plot_matrix[row$model_family, as.character(row$dataset_scope)] <- as.numeric(row$test_roc_auc)
  }

  palette_colors <- c("#4E79A7", "#F28E2B", "#59A14F", "#E15759")[seq_along(available_models)]

  png(comparison_output, width = 1800, height = 900, res = 180)
  par(mar = c(8, 5, 4, 2) + 0.1)

  barplot(
    plot_matrix,
    beside = TRUE,
    col = palette_colors,
    ylim = c(0.65, 0.95),
    ylab = "Test ROC-AUC",
    main = "Model Performance Across the Six Datasets",
    names.arg = dataset_labels,
    las = 2,
    cex.names = 0.9
  )

  legend(
    "topright",
    legend = available_models,
    fill = palette_colors,
    bty = "n",
    cex = 0.9
  )

  dev.off()
}

load_top_features <- function(source_list, top_n = 10) {
  frames <- list()

  for (source in source_list) {
    if (!file.exists(source$path)) {
      message("Skipped missing file: ", source$path)
      next
    }

    df <- read.csv(source$path, stringsAsFactors = FALSE)
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

plot_feature_importance_panels <- function(source_key, output_path, title_text) {
  frames <- load_top_features(feature_importance_sources[[source_key]])
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
  combined_df <- load_model_results()
  plot_model_performance(combined_df)

  plot_feature_importance_panels(
    source_key = "face_mask",
    output_path = face_mask_output,
    title_text = "Top XGBoost Predictors of Face Mask Behaviour"
  )

  plot_feature_importance_panels(
    source_key = "protective",
    output_path = protective_output,
    title_text = "Top XGBoost Predictors of Protective Behaviour"
  )

  message("Saved figures:")
  message(comparison_output)
  message(face_mask_output)
  message(protective_output)
}

main()
