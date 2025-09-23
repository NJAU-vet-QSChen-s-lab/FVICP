# =============================================================================
# 巨噬细胞分化轨迹分析脚本
# =============================================================================
# 功能：使用CytoTRACE方法计算细胞分化潜能，识别高/低分化潜能细胞亚群

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

# 读取巨噬细胞数据
MP <- qread(file.path(OUTPUT_DIR, "metabolic", "MP.qs"))
MP <- subset(MP, subset = celltype2 == "MP")
MP_ct <- as.matrix(MP@assays$RNA@counts)
library(CytoTRACE)
MP_results <- CytoTRACE(mat = MP_ct,  enableFast = F, 
                        ncores = 1)

plotCytoTRACE(MP_results, phenotype = MP$group_celltype3, 
              emb = MP@reductions$umap@cell.embeddings, outputDir = './')

plotCytoGenes(MP_results, numOfGenes = 10, outputDir = './')

#cyto <- read.table("CytoTRACE_plot_table.txt")
cyto <- read.table("CytoTRACE_plot_table.txt", sep = "\t", header = TRUE, stringsAsFactors = FALSE)

head(cyto)
cutoff <- quantile(cyto$CytoTRACE, 0.75)
cyto$diff <- "high"
cyto[cyto$CytoTRACE > cutoff, ]$diff <- "low"
ggplot(cyto, aes(Component1, Component2, color = diff)) +
  geom_point(size = 1.5, alpha = 1.0)

cat("=== 分化轨迹分析完成 ===\n")
