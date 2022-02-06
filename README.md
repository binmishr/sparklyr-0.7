# sparklyr-0.7

The details of the codeset and plots are included in the attached Microsoft Word Document (.docx) file in this repository. 
You need to view the file in "Read Mode" to see the contents properly after downloading the same.

A Brief Introduction - Spark Operations
=======================================

    get_spark_sql_catalog_implementation()

    Retrieve the Spark connection's SQL catalog implementation property

    spark_config()

    Read Spark Configuration

    spark_connect() spark_connection_is_open() spark_disconnect() spark_disconnect_all() spark_submit()

    Manage Spark Connections

    spark_install() spark_uninstall() spark_install_dir() spark_install_tar() spark_installed_versions() spark_available_versions()

    Download and install various versions of Spark

    spark_log()

    View Entries in the Spark Log

    spark_web()

    Open the Spark web interface

    connection_is_open()

    Check whether the connection is open

    connection_spark_shinyapp()

    A Shiny app that can be used to construct a spark_connect statement

    spark_session_config()

    Runtime configuration interface for the Spark Session

    spark_set_checkpoint_dir() spark_get_checkpoint_dir()

    Set/Get Spark checkpoint directory

    spark_table_name()

    Generate a Table Name from Expression

    spark_version_from_home()

    Get the Spark Version Associated with a Spark Installation

    spark_versions()

    Retrieves a dataframe available Spark versions that van be installed.

    spark_config_kubernetes()

    Kubernetes Configuration

    spark_config_settings()

    Retrieve Available Settings

    spark_connection_find()

    Find Spark Connection

    spark_dependency_fallback()

    Fallback to Spark Dependency

    spark_extension()

    Create Spark Extension

    spark_load_table()

    Reads from a Spark Table into a Spark DataFrame.

    list_sparklyr_jars()

    list all sparklyr-*.jar files that have been built

    spark_config_packages()

    Creates Spark Configuration

    spark_connection()

    Retrieve the Spark Connection Associated with an R Object

    spark_adaptive_query_execution()

    Retrieves or sets status of Spark AQE

    spark_advisory_shuffle_partition_size()

    Retrieves or sets advisory size of the shuffle partition

    spark_auto_broadcast_join_threshold()

    Retrieves or sets the auto broadcast join threshold

    spark_coalesce_initial_num_partitions()

    Retrieves or sets initial number of shuffle partitions before coalescing

    spark_coalesce_min_num_partitions()

    Retrieves or sets the minimum number of shuffle partitions after coalescing

    spark_coalesce_shuffle_partitions()

    Retrieves or sets whether coalescing contiguous shuffle partitions is enabled

    spark_connection-class

    spark_connection class

    spark_jobj-class

    spark_jobj class

    Spark Data
    spark_read()

    Read file(s) into a Spark DataFrame using a custom reader

    spark_read_avro()

    Read Apache Avro data into a Spark DataFrame.

    spark_read_binary()

    Read binary data into a Spark DataFrame.

    spark_read_csv()

    Read a CSV file into a Spark DataFrame

    spark_read_delta()

    Read from Delta Lake into a Spark DataFrame.

    spark_read_image()

    Read image data into a Spark DataFrame.

    spark_read_jdbc()

    Read from JDBC connection into a Spark DataFrame.

    spark_read_json()

    Read a JSON file into a Spark DataFrame

    spark_read_libsvm()

    Read libsvm file into a Spark DataFrame.

    spark_read_parquet()

    Read a Parquet file into a Spark DataFrame

    spark_read_source()

    Read from a generic source into a Spark DataFrame.

    spark_read_table()

    Reads from a Spark Table into a Spark DataFrame.

    spark_read_orc()

    Read a ORC file into a Spark DataFrame

    spark_read_text()

    Read a Text file into a Spark DataFrame

    spark_save_table()

    Saves a Spark DataFrame as a Spark table

    spark_write()

    Write Spark DataFrame to file using a custom writer

    spark_write_avro()

    Serialize a Spark DataFrame into Apache Avro format

    spark_write_orc()

    Write a Spark DataFrame to a ORC file

    spark_write_text()

    Write a Spark DataFrame to a Text file

    spark_write_csv()

    Write a Spark DataFrame to a CSV

    spark_write_delta()

    Writes a Spark DataFrame into Delta Lake

    spark_write_jdbc()

    Writes a Spark DataFrame into a JDBC table

    spark_write_json()

    Write a Spark DataFrame to a JSON file

    spark_write_parquet()

    Write a Spark DataFrame to a Parquet file

    spark_write_source()

    Writes a Spark DataFrame into a generic source

    spark_write_table()

    Writes a Spark DataFrame into a Spark table

    spark_write_rds()

    Write Spark DataFrame to RDS files

    collect_from_rds()

    Collect Spark data serialized in RDS format into R

    Spark Tables
    src_databases()

    Show database list

    tbl_cache()

    Cache a Spark Table

    tbl_change_db()

    Use specific database

    tbl_uncache()

    Uncache a Spark Table

    Spark DataFrames
    `[`(<tbl_spark>)

    Subsetting operator for Spark dataframe

    copy_to(<spark_connection>)

    Copy an R Data Frame to Spark

    dplyr_hof

    dplyr wrappers for Apache Spark higher order functions

    sdf_along()

    Create DataFrame for along Object

    sdf_bind_rows() sdf_bind_cols()

    Bind multiple Spark DataFrames by row and column

    sdf_broadcast()

    Broadcast hint

    sdf_checkpoint()

    Checkpoint a Spark DataFrame

    sdf_coalesce()

    Coalesces a Spark DataFrame

    sdf_copy_to() sdf_import()

    Copy an Object into Spark

    sdf_distinct()

    Invoke distinct on a Spark DataFrame

    sdf_drop_duplicates()

    Remove duplicates from a Spark DataFrame

    sdf_expand_grid()

    Create a Spark dataframe containing all combinations of inputs

    sdf_from_avro()

    Convert column(s) from avro format

    sdf_len()

    Create DataFrame for Length

    sdf_num_partitions()

    Gets number of partitions of a Spark DataFrame

    sdf_random_split() sdf_partition()

    Partition a Spark Dataframe

    sdf_partition_sizes()

    Compute the number of records within each partition of a Spark DataFrame

    sdf_pivot()

    Pivot a Spark DataFrame

    sdf_predict() sdf_transform() sdf_fit() sdf_fit_and_transform()

    Spark ML -- Transform, fit, and predict methods (sdf_ interface)

    sdf_rbeta()

    Generate random samples from a Beta distribution

    sdf_rbinom()

    Generate random samples from a binomial distribution

    sdf_rcauchy()

    Generate random samples from a Cauchy distribution

    sdf_rchisq()

    Generate random samples from a chi-squared distribution

    sdf_rexp()

    Generate random samples from an exponential distribution

    sdf_rgamma()

    Generate random samples from a Gamma distribution

    sdf_rgeom()

    Generate random samples from a geometric distribution

    sdf_rhyper()

    Generate random samples from a hypergeometric distribution

    sdf_rlnorm()

    Generate random samples from a log normal distribution

    sdf_rnorm()

    Generate random samples from the standard normal distribution

    sdf_rpois()

    Generate random samples from a Poisson distribution

    sdf_rt()

    Generate random samples from a t-distribution

    sdf_runif()

    Generate random samples from the uniform distribution U(0, 1).

    sdf_rweibull()

    Generate random samples from a Weibull distribution.

    sdf_read_column()

    Read a Column from a Spark DataFrame

    sdf_register()

    Register a Spark DataFrame

    sdf_repartition()

    Repartition a Spark DataFrame

    sdf_residuals()

    Model Residuals

    sdf_sample()

    Randomly Sample Rows from a Spark DataFrame

    sdf_separate_column()

    Separate a Vector Column into Scalar Columns

    sdf_seq()

    Create DataFrame for Range

    sdf_sort()

    Sort a Spark DataFrame

    sdf_to_avro()

    Convert column(s) to avro format

    sdf_with_unique_id()

    Add a Unique ID Column to a Spark DataFrame

    sdf_collect()

    Collect a Spark DataFrame into R.

    sdf_crosstab()

    Cross Tabulation

    sdf_debug_string()

    Debug Info for Spark DataFrame

    sdf_describe()

    Compute summary statistics for columns of a data frame

    sdf_dim() sdf_nrow() sdf_ncol()

    Support for Dimension Operations

    sdf_is_streaming()

    Spark DataFrame is Streaming

    sdf_last_index()

    Returns the last index of a Spark DataFrame

    sdf_save_table() sdf_load_table() sdf_save_parquet() sdf_load_parquet()

    Save / Load a Spark DataFrame

    sdf_persist()

    Persist a Spark DataFrame

    sdf_project()

    Project features onto principal components

    sdf_quantile()

    Compute (Approximate) Quantiles with a Spark DataFrame

    sdf_schema()

    Read the Schema of a Spark DataFrame

    sdf_sql()

    Spark DataFrame from SQL

    sdf_unnest_longer()

    Unnest longer

    sdf_unnest_wider()

    Unnest wider

    sdf_with_sequential_id()

    Add a Sequential ID Column to a Spark DataFrame

    inner_join(<tbl_spark>) left_join(<tbl_spark>) right_join(<tbl_spark>) full_join(<tbl_spark>)

    Join Spark tbls.

    separate

    Separate

    unite

    Unite

    nest

    Nest

    unnest

    Unnest

    pivot_wider

    Pivot wider

    pivot_longer

    Pivot longer

    fill

    Fill

    left_join

    Left join

    right_join

    Right join

    inner_join

    Inner join

    full_join

    Full join

    hof_aggregate()

    Apply Aggregate Function to Array Column

    hof_array_sort()

    Sorts array using a custom comparator

    hof_exists()

    Determine Whether Some Element Exists in an Array Column

    hof_filter()

    Filter Array Column

    hof_forall()

    Checks whether all elements in an array satisfy a predicate

    hof_map_filter()

    Filters a map

    hof_map_zip_with()

    Merges two maps into one

    hof_transform()

    Transform Array Column

    hof_transform_keys()

    Transforms keys of a map

    hof_transform_values()

    Transforms values of a map

    hof_zip_with()

    Combines 2 Array Columns

    sdf_weighted_sample()

    Perform Weighted Random Sampling on a Spark DataFrame

    transform_sdf()

    transform a subset of column(s) in a Spark Dataframe

    Spark Machine Learning
    ml_decision_tree_classifier() ml_decision_tree() ml_decision_tree_regressor()

    Spark ML -- Decision Trees

    ml_generalized_linear_regression()

    Spark ML -- Generalized Linear Regression

    ml_gbt_classifier() ml_gradient_boosted_trees() ml_gbt_regressor()

    Spark ML -- Gradient Boosted Trees

    ml_kmeans() ml_compute_cost() ml_compute_silhouette_measure()

    Spark ML -- K-Means Clustering

    ml_kmeans_cluster_eval

    Evaluate a K-mean clustering

    ml_lda() ml_describe_topics() ml_log_likelihood() ml_log_perplexity() ml_topics_matrix()

    Spark ML -- Latent Dirichlet Allocation

    ml_linear_regression()

    Spark ML -- Linear Regression

    ml_logistic_regression()

    Spark ML -- Logistic Regression

    ml_model_data()

    Extracts data associated with a Spark ML model

    ml_multilayer_perceptron_classifier() ml_multilayer_perceptron()

    Spark ML -- Multilayer Perceptron

    ml_naive_bayes()

    Spark ML -- Naive-Bayes

    ml_one_vs_rest()

    Spark ML -- OneVsRest

    ft_pca() ml_pca()

    Feature Transformation -- PCA (Estimator)

    ml_prefixspan() ml_freq_seq_patterns()

    Frequent Pattern Mining -- PrefixSpan

    ml_random_forest_classifier() ml_random_forest() ml_random_forest_regressor()

    Spark ML -- Random Forest

    ml_aft_survival_regression() ml_survival_regression()

    Spark ML -- Survival Regression

    ml_add_stage()

    Add a Stage to a Pipeline

    ml_als() ml_recommend()

    Spark ML -- ALS

    ml_approx_nearest_neighbors() ml_approx_similarity_join()

    Utility functions for LSH models

    ml_fpgrowth() ml_association_rules() ml_freq_itemsets()

    Frequent Pattern Mining -- FPGrowth

    ml_binary_classification_evaluator() ml_binary_classification_eval() ml_multiclass_classification_evaluator() ml_classification_eval() ml_regression_evaluator()

    Spark ML - Evaluators

    ml_bisecting_kmeans()

    Spark ML -- Bisecting K-Means Clustering

    ml_call_constructor()

    Wrap a Spark ML JVM object

    ml_chisquare_test()

    Chi-square hypothesis testing for categorical data.

    ml_clustering_evaluator()

    Spark ML - Clustering Evaluator

    ml_supervised_pipeline() ml_clustering_pipeline() ml_construct_model_supervised() ml_construct_model_clustering() new_ml_model_prediction() new_ml_model() new_ml_model_classification() new_ml_model_regression() new_ml_model_clustering()

    Constructors for `ml_model` Objects

    ml_corr()

    Compute correlation matrix

    ml_sub_models() ml_validation_metrics() ml_cross_validator() ml_train_validation_split()

    Spark ML -- Tuning

    ml_default_stop_words()

    Default stop words

    ml_evaluate()

    Evaluate the Model on a Validation Set

    ml_feature_importances() ml_tree_feature_importance()

    Spark ML - Feature Importance for Tree Models

    ft_word2vec() ml_find_synonyms()

    Feature Transformation -- Word2Vec (Estimator)

    is_ml_transformer() is_ml_estimator() ml_fit() ml_transform() ml_fit_and_transform() ml_predict()

    Spark ML -- Transform, fit, and predict methods (ml_ interface)

    ml_gaussian_mixture()

    Spark ML -- Gaussian Mixture clustering.

    ml_is_set() ml_param_map() ml_param() ml_params()

    Spark ML -- ML Params

    ml_isotonic_regression()

    Spark ML -- Isotonic Regression

    ft_string_indexer() ml_labels() ft_string_indexer_model()

    Feature Transformation -- StringIndexer (Estimator)

    ml_linear_svc()

    Spark ML -- LinearSVC

    ml_save() ml_load()

    Spark ML -- Model Persistence

    ml_pipeline()

    Spark ML -- Pipelines

    ml_power_iteration()

    Spark ML -- Power Iteration Clustering

    ml_stage() ml_stages()

    Spark ML -- Pipeline stage extraction

    ml_standardize_formula()

    Standardize Formula Input for `ml_model`

    ml_summary()

    Spark ML -- Extraction of summary metrics

    ml_uid()

    Spark ML -- UID

    ft_count_vectorizer() ml_vocabulary()

    Feature Transformation -- CountVectorizer (Estimator)

    Spark Feature Transformers
    ft_binarizer()

    Feature Transformation -- Binarizer (Transformer)

    ft_bucketizer()

    Feature Transformation -- Bucketizer (Transformer)

    ft_count_vectorizer() ml_vocabulary()

    Feature Transformation -- CountVectorizer (Estimator)

    ft_dct() ft_discrete_cosine_transform()

    Feature Transformation -- Discrete Cosine Transform (DCT) (Transformer)

    ft_elementwise_product()

    Feature Transformation -- ElementwiseProduct (Transformer)

    ft_index_to_string()

    Feature Transformation -- IndexToString (Transformer)

    ft_one_hot_encoder()

    Feature Transformation -- OneHotEncoder (Transformer)

    ft_quantile_discretizer()

    Feature Transformation -- QuantileDiscretizer (Estimator)

    ft_sql_transformer() ft_dplyr_transformer()

    Feature Transformation -- SQLTransformer

    ft_string_indexer() ml_labels() ft_string_indexer_model()

    Feature Transformation -- StringIndexer (Estimator)

    ft_vector_assembler()

    Feature Transformation -- VectorAssembler (Transformer)

    ft_tokenizer()

    Feature Transformation -- Tokenizer (Transformer)

    ft_regex_tokenizer()

    Feature Transformation -- RegexTokenizer (Transformer)

    ft_bucketed_random_projection_lsh() ft_minhash_lsh()

    Feature Transformation -- LSH (Estimator)

    ft_chisq_selector()

    Feature Transformation -- ChiSqSelector (Estimator)

    ft_feature_hasher()

    Feature Transformation -- FeatureHasher (Transformer)

    ft_hashing_tf()

    Feature Transformation -- HashingTF (Transformer)

    ft_idf()

    Feature Transformation -- IDF (Estimator)

    ft_imputer()

    Feature Transformation -- Imputer (Estimator)

    ft_interaction()

    Feature Transformation -- Interaction (Transformer)

    ft_max_abs_scaler()

    Feature Transformation -- MaxAbsScaler (Estimator)

    ft_min_max_scaler()

    Feature Transformation -- MinMaxScaler (Estimator)

    ft_ngram()

    Feature Transformation -- NGram (Transformer)

    ft_normalizer()

    Feature Transformation -- Normalizer (Transformer)

    ft_one_hot_encoder_estimator()

    Feature Transformation -- OneHotEncoderEstimator (Estimator)

    ft_pca() ml_pca()

    Feature Transformation -- PCA (Estimator)

    ft_polynomial_expansion()

    Feature Transformation -- PolynomialExpansion (Transformer)

    ft_r_formula()

    Feature Transformation -- RFormula (Estimator)

    ft_standard_scaler()

    Feature Transformation -- StandardScaler (Estimator)

    ft_stop_words_remover()

    Feature Transformation -- StopWordsRemover (Transformer)

    ft_vector_indexer()

    Feature Transformation -- VectorIndexer (Estimator)

    ft_vector_slicer()

    Feature Transformation -- VectorSlicer (Transformer)

    ft_word2vec() ml_find_synonyms()

    Feature Transformation -- Word2Vec (Estimator)

    ft_robust_scaler()

    Feature Transformation -- RobustScaler (Estimator)

    Spark Machine Learning Utilities
    ml_binary_classification_evaluator() ml_binary_classification_eval() ml_multiclass_classification_evaluator() ml_classification_eval() ml_regression_evaluator()

    Spark ML - Evaluators

    ml_feature_importances() ml_tree_feature_importance()

    Spark ML - Feature Importance for Tree Models

    tidy(<ml_model_als>) augment(<ml_model_als>) glance(<ml_model_als>)

    Tidying methods for Spark ML ALS

    tidy(<ml_model_generalized_linear_regression>) tidy(<ml_model_linear_regression>) augment(<ml_model_generalized_linear_regression>) augment(<ml_model_linear_regression>) glance(<ml_model_generalized_linear_regression>) glance(<ml_model_linear_regression>)

    Tidying methods for Spark ML linear models

    tidy(<ml_model_isotonic_regression>) augment(<ml_model_isotonic_regression>) glance(<ml_model_isotonic_regression>)

    Tidying methods for Spark ML Isotonic Regression

    tidy(<ml_model_lda>) augment(<ml_model_lda>) glance(<ml_model_lda>)

    Tidying methods for Spark ML LDA models

    tidy(<ml_model_linear_svc>) augment(<ml_model_linear_svc>) glance(<ml_model_linear_svc>)

    Tidying methods for Spark ML linear svc

    tidy(<ml_model_logistic_regression>) augment(<ml_model_logistic_regression>) glance(<ml_model_logistic_regression>)

    Tidying methods for Spark ML Logistic Regression

    tidy(<ml_model_multilayer_perceptron_classification>) augment(<ml_model_multilayer_perceptron_classification>) glance(<ml_model_multilayer_perceptron_classification>)

    Tidying methods for Spark ML MLP

    tidy(<ml_model_naive_bayes>) augment(<ml_model_naive_bayes>) glance(<ml_model_naive_bayes>)

    Tidying methods for Spark ML Naive Bayes

    tidy(<ml_model_pca>) augment(<ml_model_pca>) glance(<ml_model_pca>)

    Tidying methods for Spark ML Principal Component Analysis

    tidy(<ml_model_aft_survival_regression>) augment(<ml_model_aft_survival_regression>) glance(<ml_model_aft_survival_regression>)

    Tidying methods for Spark ML Survival Regression

    tidy(<ml_model_decision_tree_classification>) tidy(<ml_model_decision_tree_regression>) augment(<ml_model_decision_tree_classification>) augment(<ml_model_decision_tree_regression>) glance(<ml_model_decision_tree_classification>) glance(<ml_model_decision_tree_regression>) tidy(<ml_model_random_forest_classification>) tidy(<ml_model_random_forest_regression>) augment(<ml_model_random_forest_classification>) augment(<ml_model_random_forest_regression>) glance(<ml_model_random_forest_classification>) glance(<ml_model_random_forest_regression>) tidy(<ml_model_gbt_classification>) tidy(<ml_model_gbt_regression>) augment(<ml_model_gbt_classification>) augment(<ml_model_gbt_regression>) glance(<ml_model_gbt_classification>) glance(<ml_model_gbt_regression>)

    Tidying methods for Spark ML tree models

    tidy(<ml_model_kmeans>) augment(<ml_model_kmeans>) glance(<ml_model_kmeans>) tidy(<ml_model_bisecting_kmeans>) augment(<ml_model_bisecting_kmeans>) glance(<ml_model_bisecting_kmeans>) tidy(<ml_model_gaussian_mixture>) augment(<ml_model_gaussian_mixture>) glance(<ml_model_gaussian_mixture>)

    Tidying methods for Spark ML unsupervised models

    Extensions
    compile_package_jars()

    Compile Scala sources into a Java Archive (jar)

    connection_config()

    Read configuration values for a connection

    download_scalac()

    Downloads default Scala Compilers

    find_scalac()

    Discover the Scala Compiler

    spark_context() java_context() hive_context() spark_session()

    Access the Spark API

    hive_context_config()

    Runtime configuration interface for Hive

    invoke() invoke_static() invoke_new()

    Invoke a Method on a JVM Object

    j_invoke() j_invoke_static() j_invoke_new()

    Invoke a Java function.

    jarray()

    Instantiate a Java array with a specific element type.

    jfloat()

    Instantiate a Java float type.

    jfloat_array()

    Instantiate an Array[Float].

    register_extension() registered_extensions()

    Register a Package that Implements a Spark Extension

    spark_compilation_spec()

    Define a Spark Compilation Specification

    spark_default_compilation_spec()

    Default Compilation Specification for Spark Extensions

    spark_context_config()

    Runtime configuration interface for the Spark Context.

    spark_dataframe()

    Retrieve a Spark DataFrame

    spark_dependency()

    Define a Spark dependency

    spark_home_set()

    Set the SPARK_HOME environment variable

    spark_jobj()

    Retrieve a Spark JVM Object Reference

    spark_version()

    Get the Spark Version Associated with a Spark Connection

    Distributed Computing
    spark_apply()

    Apply an R Function in Spark

    spark_apply_bundle()

    Create Bundle for Spark Apply

    spark_apply_log()

    Log Writer for Spark Apply

    registerDoSpark()

    Register a Parallel Backend

    Livy
    livy_install() livy_available_versions() livy_install_dir() livy_installed_versions() livy_home_dir()

    Install Livy

    livy_config()

    Create a Spark Configuration for Livy

    livy_service_start() livy_service_stop()

    Start Livy

    Streaming
    stream_find()

    Find Stream

    stream_generate_test()

    Generate Test Stream

    stream_id()

    Spark Stream's Identifier

    stream_lag()

    Apply lag function to columns of a Spark Streaming DataFrame

    stream_name()

    Spark Stream's Name

    stream_read_csv()

    Read CSV Stream

    stream_read_json()

    Read JSON Stream

    stream_read_delta()

    Read Delta Stream

    stream_read_kafka()

    Read Kafka Stream

    stream_read_orc()

    Read ORC Stream

    stream_read_parquet()

    Read Parquet Stream

    stream_read_socket()

    Read Socket Stream

    stream_read_text()

    Read Text Stream

    stream_render()

    Render Stream

    stream_stats()

    Stream Statistics

    stream_stop()

    Stops a Spark Stream

    stream_trigger_continuous()

    Spark Stream Continuous Trigger

    stream_trigger_interval()

    Spark Stream Interval Trigger

    stream_view()

    View Stream

    stream_watermark()

    Watermark Stream

    stream_write_console()

    Write Console Stream

    stream_write_csv()

    Write CSV Stream

    stream_write_delta()

    Write Delta Stream

    stream_write_json()

    Write JSON Stream

    stream_write_kafka()

    Write Kafka Stream

    stream_write_memory()

    Write Memory Stream

    stream_write_orc()

    Write a ORC Stream

    stream_write_parquet()

    Write Parquet Stream

    stream_write_text()

    Write Text Stream

    reactiveSpark()

    Reactive spark reader


