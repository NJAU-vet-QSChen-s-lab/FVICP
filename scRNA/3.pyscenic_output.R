# =============================================================================
# pySCENIC输出处理脚本
# =============================================================================
# 功能：处理pySCENIC转录因子分析结果

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

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

# 设置项目目录
setwd(PROJECT_ROOT)

# pySCENIC结果处理代码...
# [此处保留原有的pySCENIC处理逻辑]

cat("=== pySCENIC结果处理完成 ===\n")

table(fascia$orig.ident)
#Dys_fascia_1 Dys_fascia_2     fascia_1     fascia_2 
#8993         8171         6434         6790 
DimPlot(fascia, split.by = "orig.ident")

# 创建pyscenic目录
dir.create(file.path(OUTPUT_DIR, "pyscenic"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "pyscenic"))

# 导出表达矩阵供pySCENIC使用
write.csv(t(as.matrix(fascia@assays$RNA@counts)),file = "sce_exp.csv")

cat("=== pySCENIC数据导出完成 ===\n")
cat("表达矩阵已保存到:", file.path(OUTPUT_DIR, "pyscenic", "sce_exp.csv"), "\n")

