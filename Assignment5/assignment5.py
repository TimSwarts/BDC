#!/usr/bin/env python3

"""
Assignment 5: Big Data Computing
"""

__author__ = "Tim Swarts"
__version__ = "0.1"

import sys
import csv
from io import StringIO
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count


# INPUT_PATH = "/homes/tswarts/jaar3.5/BDC/Assignment5/development/data/small.tsv"
INPUT_PATH = "/data/datasets/EBI/interpro/refseq_scan/bacteria.nonredundant_protein.1.protein.faa.tsv"


def create_session():
    """Create Spark session"""
    spark = (
        SparkSession.builder.master("local[16]")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return spark


def read_data(spark):
    """Read data into Spark"""
    df = spark.read.csv(INPUT_PATH, sep=r"\t", header=False)
    return df


def get_explanation(df):
    """
    This function contains the steps to fetch
    the explanation of the pipeline as a string
    Author: Tim Swarts
    :param df: Spark dataframe
    :return: explanation of the pipeline as a string
    """

    # Save the original standard output (goal is to get the explanation as a string)
    original_stdout = sys.stdout
    # Create a StringIO object
    captured_stdout = StringIO()
    # Redirect the standard output to the StringIO object
    sys.stdout = captured_stdout
    # Now when we call explain(), the output will be stored in captured_stdout
    df.explain()
    # Reset the standard output to its original value
    sys.stdout = original_stdout
    # Now we can get the explanation as a string
    df_exp = captured_stdout.getvalue()
    return df_exp


def get_unique_interpro_ids(df):
    """
    This function contains the PySpark pipeline to fetch
    the number of unique InterPro accession numbers and
    the explanation of the pipeline as a string
    :param df: Spark dataframe
    :return: number of unique InterPro accession numbers,
             explanation of the pipeline
    """

    # Get data frame in correct state
    inter_ids = df.select("_c11").distinct()

    # Get the explanation of the pipeline as a string
    inter_ids_exp = inter_ids._jdf.queryExecution().toString().split("\n\n")[3]

    # Finish processing by calling .count()
    inter_ids = inter_ids.count()

    return inter_ids, inter_ids_exp


def get_interpro_annotations_per_protein(df):
    """
    This function contains the PySpark pipeline to fetch the average number
    of InterPro annotations per protein and the explanation of the pipeline
    as a string
    :param df: Spark dataframe
    :return: average number of InterPro annotations per protein,
             explanation of the pipeline
    """

    # Get data frame in correct state
    df_with_counts = df.groupBy("_c0").agg(count(col("_c11")).alias("count_interpro"))
    # Get average
    average = df_with_counts.agg(avg(col("count_interpro"))).collect()[0][0]

    # Get the explanation of the pipeline as a string
    df_with_counts_exp = (
        df_with_counts._jdf.queryExecution().toString().split("\n\n")[3]
    )

    return average, df_with_counts_exp


def get_most_common_go_terms(df):
    """
    This function contains the PySpark pipeline to fetch the GO term occuring most often
    :param df: Spark dataframe
    :return: most occuring GO term and the explanation of the pipeline as a string
    """

    # Get data frame in correct state
    go_terms = (
        df.filter(df._c13.isNotNull() & (df._c13 != "-"))
        .groupby("_c13")
        .count()
        .sort(col("count").desc())
    )

    # Get explanation of the pipeline as a string
    go_term_exp = go_terms._jdf.queryExecution().toString().split("\n\n")[3]

    # Finish processing
    go_term = go_terms.first()[0]

    return go_term, go_term_exp


def get_size_of_an_interpro_feature(df):
    """
    This function contains the PySpark pipeline to fetch
    the size of an InterPro feature
    :param df: Spark dataframe
    :return: size of an InterPro feature and
             the explanation of the pipeline as a string
    """

    # Get data frame in correct state
    interpro_size = (
        df.filter(df._c11.isNotNull() & (df._c11 != "-"))
        .withColumn("size", df._c7 - df._c6)
        .select("size")
    )

    # Get explanation of the pipeline as a string
    interpro_size_exp = (
        interpro_size._jdf.queryExecution().toString().split("\n\n")[3]
    )

    # Finish processing
    interpro_size = interpro_size.first()[0]

    return interpro_size, interpro_size_exp


def get_top10_most_common_interpro_ids(df):
    """
    This function contains the PySpark pipeline to fetch the top 10 most common
    InterPro accession numbers.
    :param df: Spark dataframe
    :return: top 10 most common InterPro accession numbers and
             the explanation of the pipeline as a string
    """

    # Get data frame in correct state
    interpro_ids = (
        df.filter(df._c11.isNotNull() & (df._c11 != "-"))
        .groupby("_c11")
        .count()
        .sort(col("count").desc())
        .limit(10)
    )

    # Get explanation of the pipeline as a string
    interpro_ids_exp = interpro_ids._jdf.queryExecution().toString().split("\n\n")[3]

    return interpro_ids.collect(), interpro_ids_exp


def get_top10_percentage_of_proteins(df):
    """
    This function contains the PySpark pipeline to fetch the top 10
    most common InterPro accession numbers the signature of which covers
    at least 90% of the protein sequence.
    :param df: Spark dataframe
    :return: top 10 most common InterPro accession numbers and
             the explanation of the pipeline as a string
    """
    # Get data frame in correct state
    interpro_percentage_ids = (
        df.filter(df._c11.isNotNull() & (df._c11 != "-"))
        .withColumn("coverage", df._c7 - df._c6 / df._c2)
        .filter(col("coverage") > 0.9)
        .groupby("_c11").count()
        .sort(col("count").desc())
        .limit(10)
    )

    # Get explanation of the pipeline as a string
    interpro_percentage_ids_exp = (
        interpro_percentage_ids._jdf.queryExecution().toString().split("\n\n")[3]
    )

    return interpro_percentage_ids.collect(), interpro_percentage_ids_exp


def get_question_7(df):
    """
    This function contains the PySpark pipeline to fetch the answer to question 7
    :param df: Spark dataframe
    :return: answer to question 7 and the explanation of the pipeline as a string
    """

    # Get data frame in correct state
    question_7 = (
        df.filter(df._c12.isNotNull() & (df._c12 != "-"))
        .groupby("_c12").count()
        .sort(col("count").desc())
        .limit(10)
    )

    # Get explanation of the pipeline as a string
    question_7_exp = question_7._jdf.queryExecution().toString().split("\n\n")[3]

    return question_7.collect(), question_7_exp


def get_question_8(df):
    """
    This function contains the PySpark pipeline to fetch the answer to question 8
    :param df: Spark dataframe
    :return: answer to question 7 and the explanation of the pipeline as a string
    """

    # Get data frame in correct state
    question_8 = (
        df.filter(df._c12.isNotNull() & (df._c12 != "-"))
        .groupby("_c12").count()
        .sort(col("count").asc())
        .limit(10)
    )

    # Get explanation of the pipeline as a string
    question_8_exp = question_8._jdf.queryExecution().toString().split("\n\n")[3]

    return question_8.collect(), question_8_exp


def get_question_9(df):
    """
    This function contains the PySpark pipeline to fetch the answer to question 9
    :param df: Spark dataframe
    :return: answer to question 9 and the explanation of the pipeline as a string
    """

    # Get data frame in correct state
    question_9 = (
        df.withColumn("coverage", df._c7 - df._c6 / df._c2)
        .filter((col("coverage") > 0.9) & (df._c12 != "-"))
        .groupby("_c12").count()
        .sort(col("count").desc()).limit(10)
    )

    # Get explanation of the pipeline as a string
    question_9_exp = question_9._jdf.queryExecution().toString().split("\n\n")[3]

    return question_9.collect(), question_9_exp


def get_R2_between_size_and_interpro_ids(df):
    """
    This function contains the PySpark pipeline to fetch the R2 between the size
    of a protein and the number of InterPro annotations it has.
    :param df: Spark dataframe
    :return: R2 between the size of a protein and the number of InterPro annotations it
             has and the explanation of the pipeline as a string
    """

    # Get data frame in correct state
    grouped_df = (
        df.filter(df._c11.isNotNull() & (df._c11 != "-"))
        .groupby("_c11")
        .agg(avg("_c2").alias("avg_size"), count("_c11").alias("ïnterpro_count"))
    )

    # Get explanation of the pipeline as a string
    grouped_df_exp = grouped_df._jdf.queryExecution().toString().split("\n\n")[3]

    # Get R2 between the size of a protein and the number of InterPro annotations it has
    r2 = grouped_df.stat.corr("avg_size", "ïnterpro_count") ** 2

    return r2, grouped_df_exp


def main():
    """Main function"""
    # Create Spark session
    spark = create_session()
    # Read data into Spark
    df = read_data(spark)

    # Question 1: How many unique InterPro IDs are there?
    numbr_ids, numbr_ids_exp = get_unique_interpro_ids(df)

    # Question 2: How many InterPro annotations are there per protein?
    (
        interpro_annotations_per_protein,
        average_exp,
    ) = get_interpro_annotations_per_protein(df)

    # Question 3: What are the most common GO terms?
    most_occuring_go_term, go_term_exp = get_most_common_go_terms(df)

    # Question 4: What is the size of an InterPro feature?
    interpro_size, interpro_size_exp = get_size_of_an_interpro_feature(df)

    # Question 5: What are the top 10 most common InterPro IDs?
    interpro_ids, interpro_ids_exp = get_top10_most_common_interpro_ids(df)

    # Qestion 6: What are the top 10 most common InterPRO accesions that span >90% of the protein?
    interpro_percentage_ids, interpro_percentage_ids_exp = get_top10_percentage_of_proteins(df)

    # Question 7: I honestly don't undestands this quesion, I think I need groupby and count _c12
    question_7, question_7_exp = get_question_7(df)

    # Question 8: inverse of question 7
    question_8, question_8_exp = get_question_8(df)

    # Question 9: combine question 6 and 7
    question_9, question_9_exp = get_question_9(df)

    # Question 10: What is the R^2 coffiecient between the size of a protein and the number of InterPro annotations?
    r2, r2_exp = get_R2_between_size_and_interpro_ids(df)

    # Write results to csv file
    with open("output.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, dialect="excel", delimiter=",", quotechar='"')

        csvwriter.writerow(["1", numbr_ids, rf"{numbr_ids_exp}"])
        csvwriter.writerow(["2", interpro_annotations_per_protein, rf"{average_exp}"])
        csvwriter.writerow(["3", most_occuring_go_term, rf"{go_term_exp}"])
        csvwriter.writerow(["4", interpro_size, rf"{interpro_size_exp}"])
        csvwriter.writerow(["5", interpro_ids, rf"{interpro_ids_exp}"])
        csvwriter.writerow(["6", interpro_percentage_ids, rf"{interpro_percentage_ids_exp}"])
        csvwriter.writerow(["7", question_7, rf"{question_7_exp}"])
        csvwriter.writerow(["8", question_8, rf"{question_8_exp}"])
        csvwriter.writerow(["9", question_9, rf"{question_9_exp}"])
        csvwriter.writerow(["10", r2, rf"{r2_exp}"])

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
