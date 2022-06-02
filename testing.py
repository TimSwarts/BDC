words = ['hello', 'world', 'you', 'look', 'nice']
new = ", ".join(f"'{w}'" for w in words)

# data
data = {'chromosome': 'chr1', 'reference': 'C', 'RefSeq_Gene': 'SKI/MORN1', 'RefSeq_Func': 'intergenic', 'dbsnp138': 'rs2843156', '1000g2015aug_EUR': '0.8211', 'LJB2_SIFT': '', 'LJB2_PolyPhen2_HDIV': '', 'LJB2_PolyPhen2_HVAR': '', 'CLINVAR': ''}
data_columns = ['chromosome', 'RefSeq_Gene']

# empty string
query_part_two = ", ".join(f"'{data[k]}'" for k in data_columns)
# loop
# for column in data_columns:
#     query_part_two += ", "f"'{data[column]}', "

# strip trailing comma
# query_part_two = query_part_two.rstrip(", ")
print(query_part_two)



