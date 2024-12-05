# data1 = open("pt_data_a6000_flash_attn1.tsv", "r").readlines()
# data2 = open("pt_data_a6000_flash_attn2.tsv", "r").readlines()

data1 = open("pt_data_a100_flash_attn1.tsv", "r").readlines()
data2 = open("pt_data_a100_flash_attn2.tsv", "r").readlines()

header = data1[0].strip().split("\t")
new_header = header[:5] + ["Flash Attention1 (ms)"
                           ] + ["Flash Attention2 (ms)"] + header[6:]

# with open("MHA_A6000.tsv", "w") as fdata:
with open("MHA_A100.tsv", "w") as fdata:
    fdata.write("%s\n" % ("\t".join(new_header)))
    for i in range(len(data1) - 1):
        items1 = data1[i + 1].strip().split("\t")
        items2 = data2[i + 1].strip().split("\t")
        ft2 = items2[5]
        new_items = items1[0:6] + [ft2] + items2[6:]
        fdata.write("%s\n" % "\t".join(new_items))
