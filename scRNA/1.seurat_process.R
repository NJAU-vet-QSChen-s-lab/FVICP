# =============================================================================
# Seurat标准流程脚本
# =============================================================================
# 功能：合并正常与疾病样本，进行Seurat标准流程分析

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

# 读取质控后的数据
normal <- qread(file.path(OUTPUT_DIR, "qc", "normal_fascia.qs"))
Dys <- qread(file.path(OUTPUT_DIR, "qc", "Dys_fascia.qs"))

# 合并样本
fascia <- merge(normal, y = Dys)
qsave(fascia, file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

#-------------------------------------------------------------------------------
# Seurat标准流程
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

# 读取合并后的数据
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

# 执行Seurat标准流程
fascia <- seurat_process(fascia, workers1 = 2, workers2 = 2, do_harmony = FALSE, maxmem = 8)
DimPlot(fascia, split.by = "orig.ident")

# 保存处理后的对象
qsave(fascia, file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

cat("=== Seurat标准流程完成 ===\n")
cat("处理后的细胞数:", ncol(fascia@assays$RNA@counts), "\n")
cat("处理后的基因数:", nrow(fascia@assays$RNA@counts), "\n")
