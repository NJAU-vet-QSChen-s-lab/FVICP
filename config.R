# =============================================================================
# 配置文件：路径设置
# =============================================================================
# 注意：请根据您的实际环境修改以下路径

# 项目根目录
PROJECT_ROOT <- "."

# 环境文件目录
ENV_DIR <- file.path(PROJECT_ROOT, "envs")

# 数据目录
DATA_DIR <- file.path(PROJECT_ROOT, "data")

# 输出目录
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output")

# 质控参数设置
QC_PARAMS <- list(
  # 疾病样本质控参数
  dys_fascia = list(
    nCount_RNA_max = 28804.39,
    nFeature_RNA_max = 6259.611,
    percent_mt_max = 12.00886
  ),
  # 正常样本质控参数
  normal_fascia = list(
    nCount_RNA_max = 31017.95,
    nFeature_RNA_max = 7201.482,
    percent_mt_max = 10.37826
  ),
  # 通用质控参数
  general = list(
    min_cells = 3,
    min_features = 50,
    mt_pattern = "^Mt-"  # 小鼠线粒体基因模式
  )
)

# 创建必要的目录
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "qc"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "seurat"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "metabolic"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "cellcall"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "pytorch"), showWarnings = FALSE, recursive = TRUE)

# 打印配置信息
cat("=== 项目配置信息 ===\n")
cat("项目根目录:", PROJECT_ROOT, "\n")
cat("环境文件目录:", ENV_DIR, "\n")
cat("数据目录:", DATA_DIR, "\n")
cat("输出目录:", OUTPUT_DIR, "\n")
cat("==================\n") 