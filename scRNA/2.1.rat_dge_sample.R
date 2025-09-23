# =============================================================================
# 样本差异表达分析脚本
# =============================================================================
# 功能：按样本进行差异表达分析

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

# 创建DGEs目录
dir.create(file.path(OUTPUT_DIR, "DGEs"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "DGEs"))

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))
table(fascia$orig.ident)
#Dys_fascia_1 Dys_fascia_2     fascia_1     fascia_2 
#8993         8171         6434         6790 
DimPlot(fascia, split.by = "orig.ident")

fascia$sample <- recode(fascia$orig.ident, 
                        "fascia_1" = "health", 
                        "fascia_2" = "health", 
                        "Dys_fascia_1" = "dysentery", 
                        "Dys_fascia_2" = "dysentery")
table(fascia$sample)

workers1 = 10
cl <- detectCores()
makecore(workers1, 64)
cl <- makeCluster(workers1)
fascia@active.ident <- as.factor(fascia$sample)
dge <- FindMarkers(fascia, group.by = "sample", ident.1 = "dysentery", ident.2 = "health", verbose = T, 
                   logfc.threshold = 0, min.pct = 0)
write.csv(dge, "dges_sample.csv")
plan("sequential")
stopCluster(cl)
rm(cl)

cat("=== 样本差异表达分析完成 ===\n")
cat("结果已保存到:", file.path(OUTPUT_DIR, "DGEs"), "\n")

