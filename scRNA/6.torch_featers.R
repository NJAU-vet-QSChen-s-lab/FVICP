# =============================================================================
# 深度学习特征整合分析脚本
# =============================================================================
# 功能：结合深度学习特征重要性与DGE结果，筛选关键基因

print("Hello world!")
rm(list = ls())

# 加载配置文件
source("../config.R")

# 设置工作目录
setwd(ENV_DIR)
lf <- list.files("./")
for (file in lf) {
  source(file)
}
rm(lf, file)

# 设置项目目录
setwd(PROJECT_ROOT)

# 创建torch_features目录
torch_features_dir <- file.path(OUTPUT_DIR, "torch_features")
dir.create(torch_features_dir, showWarnings = FALSE, recursive = TRUE)
setwd(torch_features_dir)

# 加载必要的包
library(RImagePalette)
library(imager)
library(scales)
library(ggplot2)
library(ggprism)
library(reshape)
library(ggalluvial)
library(plotly)
library(paletteer)
library(foreach)
library(doParallel)

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

#===============================================================================
# 数据预处理和分组
#===============================================================================

cat("=== 开始数据预处理 ===\n")
table(fascia$celltype)

# 移除Others和MuSC细胞类型
fascia <- subset(fascia, subset = celltype %in% c("Others", "MuSC"), invert = T)

# 按样本分割并创建group_celltype标识
sclist <- SplitObject(fascia, split.by = "sample")
namelist <- names(sclist)

for (i in 1:length(sclist)) {
  lt <- length(table(sclist[[i]]$celltype))
  for (j in lt) {
    sclist[[i]]$group_celltype <- paste(namelist[[i]], "_", sclist[[i]]$celltype, sep = "")
  }
  cat("Sample", namelist[i], "group_celltype table:\n")
  print(table(sclist[[i]]$group_celltype))
}

# 合并数据并更新group_celltype
test <- merge(sclist[[1]], y = sclist[[2]])
fascia$group_celltype <- test$group_celltype
rm(test, sclist)

cat("Final celltype distribution:\n")
print(table(fascia$celltype))

# 设置active.ident为sample
fascia@active.ident <- as.factor(fascia$sample)

#===============================================================================
# 差异基因表达分析 (DGE)
#===============================================================================

cat("=== 开始差异基因表达分析 ===\n")

# 按细胞类型分割
sclist <- SplitObject(fascia, split.by = "celltype")
namelist <- names(sclist)
cat("Cell types for analysis:", paste(namelist, collapse = ", "), "\n")

# 设置并行环境
numCores <- detectCores()
cl <- makeCluster(numCores)
registerDoParallel(cl)

cat("Running DGE analysis in parallel...\n")

# 并行计算差异基因表达
DGElist <- foreach(sce=sclist, .combine=list, .multicombine=TRUE, .packages = "Seurat") %dopar% {
  FindMarkers(sce, group.by = "sample", ident.1 = "dysentery", ident.2 = "health", 
              logfc.threshold = 0, min.cells.feature = 0, min.cells.group = 0)
}

# 停止并行环境
stopCluster(cl)

# 保存DGE结果
for (i in seq_along(sclist)) {
  filename <- paste0(namelist[i], "_fascia_dge_dys_vs_health.csv")
  write.csv(DGElist[[i]], file = filename)
  cat("Saved DGE results for", namelist[i], "to", filename, "\n")
}

#===============================================================================
# 深度学习特征重要性与DGE结果整合
#===============================================================================

cat("=== 开始特征重要性与DGE结果整合 ===\n")

# 读取SHAP特征重要性文件
SHAPfiles <- list.files(pattern = "^feature_importance")
cat("Found SHAP files:", paste(SHAPfiles, collapse = ", "), "\n")

SHAPfiles <- lapply(SHAPfiles, function(file){
  read_csv(file)[, -1]
})
names(SHAPfiles) <- list.files(pattern = "^feature_importance")

# 读取DGE结果文件
DGEfiles <- list.files("./", pattern = "dys_vs_health.csv$")
cat("Found DGE files:", paste(DGEfiles, collapse = ", "), "\n")

DGEfiles <- lapply(DGEfiles, function(file){
  read_csv(file)
})
names(DGEfiles) <- list.files("./", pattern = "dys_vs_health.csv$")

# 定义细胞类型映射
celltype_mapping <- list(
  "feature_importance_faCPTC.csv" = "CPTC_fascia_dge_dys_vs_health.csv",
  "feature_importance_faMC.csv" = "MC_fascia_dge_dys_vs_health.csv",
  "feature_importance_faEC.csv" = "EC_fascia_dge_dys_vs_health.csv",
  "feature_importance_faETC.csv" = "ETC_fascia_dge_dys_vs_health.csv",
  "feature_importance_faFC.csv" = "FC_fascia_dge_dys_vs_health.csv",
  "feature_importance_faLP.csv" = "LP_fascia_dge_dys_vs_health.csv",
  "feature_importance_faMuC.csv" = "MuC_fascia_dge_dys_vs_health.csv"
)

# 处理每个细胞类型的特征整合
for (shap_name in names(celltype_mapping)) {
  if (shap_name %in% names(SHAPfiles) && celltype_mapping[[shap_name]] %in% names(DGEfiles)) {
    
    cat("Processing", shap_name, "with", celltype_mapping[[shap_name]], "\n")
    
    # 获取对应的SHAP和DGE数据
    SHAPfile <- SHAPfiles[[shap_name]]
    DGEfile <- DGEfiles[[celltype_mapping[[shap_name]]]]
    
    # 确保DGE文件有正确的列名
    colnames(DGEfile)[1] <- "feature"
    
    # 合并SHAP和DGE数据
    cbfile <- left_join(SHAPfile, DGEfile, by = "feature")
    
    # 筛选条件：SHAP值 > 0 且 p值 > 0
    cbfile <- subset(cbfile, subset = shap_value > 0 & p_val > 0)
    
    # 保存结果
    output_name <- paste0("MLP_filter_", gsub("feature_importance_fa(.+)\\.csv", "\\1", shap_name), "_fascia_dge_dys_vs_health.csv")
    write.csv(cbfile, output_name)
    cat("Saved filtered results to", output_name, "\n")
  }
}

#===============================================================================
# GSEA富集分析
#===============================================================================

cat("=== 开始GSEA富集分析 ===\n")

# 读取MLP筛选后的文件
MLP_files <- list.files(pattern = "^MLP_filter")
cat("Found MLP filter files:", paste(MLP_files, collapse = ", "), "\n")

MLP_files <- lapply(MLP_files, function(file){
  read_csv(file)[, -1]
})
names(MLP_files) <- list.files(pattern = "^MLP_filter")

# 获取MLP_files的名称列表
namelist <- names(MLP_files)

# 遍历每个MLP文件进行GSEA分析
for (i in 1:length(MLP_files)) {
  cat("Processing GSEA for", namelist[i], "\n")
  
  # 获取当前数据
  current_list <- MLP_files[[i]]
  
  # 准备GSEA输入数据
  gsedf <- current_list$avg_log2FC
  names(gsedf) <- as.character(current_list$feature)
  gsedf <- sort(gsedf, decreasing = TRUE)  # 排序很重要
  
  # 进行GSEA分析
  tryCatch({
    gse_BP <- gseGO(gsedf, ont = "BP", OrgDb = 'org.Rn.eg.db', 
                    pvalueCutoff = 1, keyType = "SYMBOL")
    gse_bp <- data.frame(gse_BP)
    
    # 保存结果
    csv_name <- paste0("GSEA_BP_", namelist[i], ".csv")
    rds_name <- paste0("GSEA_BP_", namelist[i], ".rds")
    
    write.csv(gse_bp, csv_name)
    saveRDS(gse_BP, rds_name)
    
    cat("Saved GSEA results for", namelist[i], "\n")
  }, error = function(e) {
    cat("Error in GSEA analysis for", namelist[i], ":", e$message, "\n")
  })
}

#===============================================================================
# 分析完成
#===============================================================================

# 返回项目根目录
setwd(PROJECT_ROOT)

cat("=== 深度学习特征整合分析完成 ===\n")
cat("输出目录:", torch_features_dir, "\n")
cat("生成的文件包括:\n")
cat("- DGE分析结果文件\n")
cat("- MLP筛选后的特征文件\n") 
cat("- GSEA富集分析结果\n")
