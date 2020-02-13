import tensorflow as tf



#def basic_params1():
class basic_params1():
  """A set of basic hyperparameters."""
  #return tf.contrib.training.HParams(
  def __init__(self):
      # If the problem consists of variable-length sequences
      # (see problem.batch_size_means_tokens()), then this is the number
      # of tokens per batch per GPU or per TPU core.  Otherwise, this is
      # the number of examples per GPU or per TPU core.
      self.batch_size=4096,
      # If True, then if the features are of variable length, the batch_size is
      # used as the actual batch size (and not tokens per batch).
      self.use_fixed_batch_size=False,
      self.num_hidden_layers=4,
      self.kernel_height=3,
      self.kernel_width=1,
      self.hidden_size=64,
      self.compress_steps=0,
      # All hyperparameters ending in "dropout" are automatically set to 0.0
      # when not in training mode.
      self.dropout=0.2,
      self.clip_grad_norm=2.0,
      self.grad_noise_scale=0.0,
      self.summarize_grads=False,
      # Whether to log the name and size of every variable
      self.summarize_vars=False,
      self.initializer="orthogonal",
      self.initializer_gain=1.5,
      self.label_smoothing=0.1,
      self.optimizer="Adam",
      self.optimizer_adam_epsilon=1e-6,
      self.optimizer_adam_beta1=0.85,
      self.optimizer_adam_beta2=0.997,
      self.optimizer_momentum_momentum=0.9,
      self.optimizer_momentum_nesterov=False,
      self.optimizer_adafactor_beta1=0.0,
      self.optimizer_adafactor_beta2=0.999,
      self.optimizer_adafactor_factored=True,
      self.optimizer_adafactor_decay_type="pow",
      self.optimizer_adafactor_memory_exponent=0.8,
      self.optimizer_adafactor_clipping_threshold=1.0,
      self.optimizer_adafactor_multiply_by_parameter_scale=True,
      # Number of accumulating steps for multi step optimizers.
      self.optimizer_multistep_accumulate_steps=None,
      self.weight_decay=1e-6,
      self.weight_noise=0.0,
      # Defines the learning rate as a product of named functions.
      # Available functions are listed in learning_rate._LEARNING_RATE_FUNCTIONS
      # e.g. "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size"
      self.learning_rate_schedule="legacy",
      self.learning_rate_constant=1.0,
      # If learning_rate_schedule=="legacy",
      # then we specify decay scheme here.  Warmup is always exponential,
      # except with "noam" learning rate decay scheme.
      # see optimize.legacy_learning_rate_schedule()
      # TODO(noam): migrate everyone away from this.
      self.learning_rate_decay_scheme="none",
      # decay_steps and decay_staircase for learning_rate_decay_scheme=="exp"
      self.learning_rate_decay_steps=5000,
      self.learning_rate_decay_staircase=False,
      self.learning_rate_minimum=None,
      self.learning_rate_decay_rate=1.0,
      self.learning_rate_warmup_steps=100,
      self.learning_rate_cosine_cycle_steps=250000,
      self.learning_rate=0.1,
      self.sampling_method="argmax",  # "argmax" or "random"
      self.sampling_temp=1.0,  # temperature for sampling
      # expand the logits a piece at a time - saves memory.
      self.factored_logits=False,
      self.multiply_embedding_mode="sqrt_depth",
      # Parameters related to mixtures of experts.
      self.moe_hidden_sizes="2048",  # hidden layer sizes (comma-separated)
      self.moe_num_experts=64,  # number of experts per layer
      self.moe_k=2,  # how many experts to use for each batch element
      self.moe_loss_coef=1e-2,
      # Sequences of operations to perform on layer input and layer output.
      # Used by common_layers.layer_preprocess, common_layers.layer_postprocess
      # Each character represents an operation:
      # none: no preprocessing
      #    d: apply dropout
      #    n: apply normalization (see norm_type and norm_epsilon)
      #    a: add layer input (residual connection - only during postprocess)
      # The special string "none" is used instead of the empty string
      # to indicate no pre/postprocessing, since the empty string causes
      # trouble for hyperparameter tuning.
      # TODO(noam): The current settings ("", "dan") are the published version
      # of the transformer.  ("n", "da") seems better for harder-to-learn
      # models, so it should probably be the default.
      self.layer_preprocess_sequence="none",
      self.layer_postprocess_sequence="dan",
      # dropout rate to use during layer_preprocess and layer_postprocess
      self.layer_prepostprocess_dropout=0.1,
      # broadcast dimensions for layer_prepostprocess_dropout
      # a comma-separated list of integers.
      # see common_layers.dropout_with_broadcast_dims()
      # Change this to "1" to save memory.
      self.layer_prepostprocess_dropout_broadcast_dims="",
      # dropout some symbols (set them to 0) before embedding.
      self.symbol_dropout=0.0,
      # What type of normalization to use
      self.norm_type="layer",  # "batch", layer", "noam", "none".
      # epsilon parameter to normalization function
      self.norm_epsilon=1e-6,
      self.symbol_modality_num_shards=1,
      # pad vocabularies so that this value divides the vocabulary size.
      self.vocab_divisor=1,
      # During training, we drop sequences whose inputs and targets are shorter
      # than min_length
      self.min_length=0,
      # During training, we drop sequences whose inputs or targets are longer
      # than max_length.
      # If max_length==0, we use hparams.batch_size instead.
      self.max_length=0,
      # Maximum length in the smallest length bucket.  Setting this
      # flag too high will result in wasteful padding of short
      # sequences.  Due to some (hopefully) temporary hacks in the
      # data reading and batching code, setting this flag too low
      # results in a very long batch-shuffling queue.
      # TODO(noam): change this once the Datasets API changes.
      self.min_length_bucket=8,
      # This flag controls the number of length buckets in the data
      # reader.  The buckets have maximum lengths from
      # min_bucket_length to (max_length or batch_size), increasing
      # (approximately) by factors of length_bucket_step.
      self.length_bucket_step=1.1,
      # If set to True, drop sequences longer than max_length during eval.
      # This affects the validity of the evaluation metrics.
      self.eval_drop_long_sequences=False,
      # If True, run the model autoregressively instead of teacher-forcing
      # during eval
      self.eval_run_autoregressive=False,
      # TODO(lukaszkaiser): these parameters should probably be set elsewhere.
      # (SymbolModality) - If this flag is on, we try to share all of the input
      # embeddings, the target embeddings and the softmax weights.
      self.shared_embedding_and_softmax_weights=False,
      # (SymbolModality) - If this flag is on, we try to share the input
      # embeddings and the target embeddings.
      # You can also share the input embeddings with the target embeddings
      # by using a problem_hparams that uses the same modality object for
      # the input_modality and target_modality.
      self.shared_embedding=False,
      # In SymbolModality, skip the top layer, assume we're providing logits.
      self.symbol_modality_skip_top=False,
      # For each feature for which you want to override the default input
      # modality, add an entry to this semicolon-separated string. Entries are
      # formatted "feature_name:modality_type:modality_name", e.g.
      # "inputs:symbol:default;other_inputs:audio:identity".
      self.input_modalities="default",  # We don't use empty string in params.
      # To override the default target modality, specify
      # "modality_type:modality_name", e.g. "symbol:ctc".
      self.target_modality="default",
      # The maximum length of "input" sequence.
      # Sequences longer than this value will be truncated. 0 or negative values
      # mean there is no maximum or truncation.
      # You can change this behavior by overriding preprocess_example() method
      # in your problem class.
      self.max_input_seq_length=0,
      # The maximum length of "target" sequence.
      # Sequences longer than this value will be truncated. 0 or negative values
      # mean there is no maximum or truncation.
      # You can change this behavior by overriding preprocess_example() method
      # in your problem class.
      self.max_target_seq_length=0,
      # if nonzero, we split the target sequences on example read.
      # This is for use with language modeling problems with fixed length
      # examples.  e.g.  The examples may be written with length 65536, but we
      # want to split each example into 64 examples of length 1024.
      self.split_to_length=0,
      # Video settings: how many frames to batch on input and targets.
      self.video_num_input_frames=1,
      self.video_num_target_frames=1,
      # This flag allows us to optionally treat a seq-to-seq problem
      # as a language model.  Legal values are:
      #
      # "none" - Do not prepend the inputs to the targets.
      # "prepend_inputs_masked_attention"
      #     replace "targets" in preprocessing with
      #     tf.concat([inputs, [0], targets], axis=1)
      #     i.e. we prepend the inputs to the targets with a single
      #     padding token in between.  Use masked self-attention on the
      #     entire resulting sequence.  During training, we compute losses on
      #     the combined sequence.  During eval, we compute the metrics
      #     on only the targets portion.
      # "prepend_inputs_full_attention"
      #     similar to the previous option except that each
      #     position in the inputs portion can see the
      #     entire inputs portion.  This removes the challenge of
      #     autoregressively predicting the inputs portion.
      self.prepend_mode="none",
      # Scheduled sampling is interesting for auto-regressive models.
      # It runs an additional step using the generated output as autoregressive
      # targets, which can improve the models inference results later. The
      # parameter scheduled_sampling_prob determines with what probability
      # will such additional step be run. It's turned off (0.0) by default.
      # This probability will exponentially warm up for the number of
      # steps determined by scheduled_sampling_warmup_steps.
      # The tensor used for the second step will consist of outputs from
      # the first step mixed with gold truth, with the proportion of gold
      # determined by scheduled_sampling_gold_mixin_prob.
      self.scheduled_sampling_prob=0.0,
      self.scheduled_sampling_warmup_steps=50000,
      self.scheduled_sampling_gold_mixin_prob=0.5,
      # This setting controls whether to copy variables around in a daisy chain
      # (if true) or leave their placement to TensorFlow. It only affects multi
      # device training and mostly should be turned on for performance. One
      # exception are recurrent models: with dynamic loops it must be off.
      self.daisy_chain_variables=True,
      # If True in PREDICT mode, then last-position-only optimizations are not
      # used.
      self.force_full_predict=False,
      # Set this for pure model parallelism.  There is only one data shard.
      self.no_data_parallelism=False,
      # dtype used for activations. - "float32" or "bfloat16"
      # activation_dtype="bfloat16" currently only works on TPU.
      #    It lowers activation-memory usage
      #    and does not appear to affect quality.
      #    You can train on TPU with activation_dtype="bfloat16" and evaluate
      #    on CPU/GPU with activation_dtype="float32"
      self.activation_dtype="float32",
      # dtype used for parameters: "float32" or "bfloat16"
      # bfloat16 currently only works with optimizer="adafactor".
      #   The savings in memory allow for training larger models.
      #   Weights are encoded as (w*128)^8, using pseudostochastic
      #   roundoff.  Initial experiments show that model quality is similar
      #   to baseline for about 3M training steps, but worse thereafter.
      self.weight_dtype="float32",
      # Directory containing a checkpoint for a pretrained model. This will only
      # be used if a new run is being started. Parameters not found in the
      # pretrained model will be randomly initialized. Superfluous parameters in
      # the pretrained model will be ignored.
      self.pretrained_model_dir="",
      # Threshold used for two cases: the primary task probability for the
      # constant mixing schedule, and the exponential schedule limit for when
      # mixing should stop (eg: 0.5 means stop at 50-50 mixing, 0.8 means stop
      # at 20-80 mixing for the primary-others mixing case.)
      self.multiproblem_schedule_threshold=0.5,
      # The number of examples at which the proportion of the mixed in datasets
      # is multiproblem_schedule_threshold
      self.multiproblem_schedule_max_examples=1e7,
      # When training multiproblems, we can mix the data according to different
      # schedules. Example: a constant schedule mixing 20-80 between the primary
      # and other tasks.
      # A list of supported schedules can be found in
      # `data_generators.multi_problem.py`.
      self.multiproblem_mixing_schedule="constant"
  
  
  
  
if __name__=='__main__':
    hp=basic_params1()
    hp.hidden_size=13
    print ('')