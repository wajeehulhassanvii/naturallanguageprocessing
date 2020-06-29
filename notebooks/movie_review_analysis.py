
"""
This script will take file containing reviews of the same.
It will output the results of analysis to std.out.
"""

import argparse as ap
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

if __name__ == '__main__':
    print('begining...')
    parser = ap.ArgumentParser(description='Movie Review Analysis')
    parser.add_argument('-file', metavar='DATA', type=str,
                       required=True, help='The file containing the\
                       reviews in JSON format, one JSON review\
                       per line')
    
    options = vars(parse.parse_args())
    
    spark = SparkSession.builder\
        .appName("Movie Analysis")\
        .config("spark.driver.memory", "12g")\
        .config("spark.executor.memory", "12g")\
        .config("spark.jars.packages",
               "JohnSnowLabs:spark-nlp:2.2.2")\
        .getOrCreate()
    
    nlp_pipeline = PipelineModel.load('../checkpoints/nlp_pipeline.3.12')
    model = PipelineModel.load('../checkpoints/model.3.12/')
    
    data = spark.read.json(options['file'])
    
    nlp_procd = nlp_pipeline.transform(data)
    preds = model.transform(nlp_procd)
    
    results = preds.selectExpr(
        'count(*)',
        'mean(rawPredictions[1])',
        'std(rawPredictions[1])',
        'median(rawPredictions[1])',
        'min(rawPredictions[1])',
        'max(rawPredictions[1])'
    ).first().asDict()
    
    print(json.dump(results))
