from typing import Dict, List, Tuple, Optional, Union

import tensorflow as tf
import tensorflow_addons as tfa
from utils import entmax15


class GLULayer(tf.keras.layers.Layer):
    def __init__(
            self,
            units: int = 16,
            fc_layer: Optional[tf.keras.layers.Dense] = None,
            virtual_batch_size: Optional[int] = None,
            momentum: float = 0.98,
            **kwargs
    ):
        """
        Creates a layer with a fully-connected linear layer, followed by batch
        normalization, and a gated linear unit (GLU) as the activation function.

        Parameters:
        -----------
        units: int
            Number of units in layer. Default (16).
        fc_layer:tf.keras.layers.Dense
            This is useful when you want to create a GLU layer with shared parameters. This
            is necessary because batch normalization should still be uniquely initialized
            due to the masked inputs in TabNet steps being in a different scale than the
            original input. Default (None) creates a new FC layer.
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller
            than and a factor of the overall batch size. Default (None) runs regular batch
            normalization. If an integer value is specified, GBN is run with that virtual
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values
            correspond to larger impact of batch on the rolling statistics computed in
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(GLULayer, self).__init__(**kwargs)
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

        if fc_layer:
            self.fc = fc_layer
        else:
            self.fc = tf.keras.layers.Dense(self.units * 2, use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size,
                                                     momentum=self.momentum)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.math.multiply(x[:, :self.units], tf.nn.sigmoid(x[:, self.units:]))
        return x


class FeatureTransformer(tf.keras.layers.Layer):
    def __init__(
            self,
            n_dependent_glus: int = 2,
            shared_glu_fc_layers: Optional[List[tf.keras.layers.Dense]] = None,
            units: int = 16,
            virtual_batch_size: Optional[int] = None,
            momentum: float = 0.98,
            **kwargs
    ):
        """
        Creates a feature transformer for non-linear processing of features.

        Parameters:
        -----------
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing
            the number of step-dependent layers is an effective strategy to improve predictive
            performance. Default (2).
        shared_glu_fc_layers: List[tf.keras.layers.Dense]
            A list of dense layers to construct shared GLU layers. Default (None) creates only
            n_dependent_glus dependent GLU layers and no shared layers. Total number of GLU layers
            in this feature transformer is len(shared_glu_layers) + n_dependent_glus.
        units: int
            Number of units in each GLU layer. Default (16).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller
            than and a factor of the overall batch size. Default (None) runs regular batch
            normalization. If an integer value is specified, GBN is run with that virtual
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values
            correspond to larger impact of batch on the rolling statistics computed in
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(FeatureTransformer, self).__init__(**kwargs)
        n_glu_layers = (len(shared_glu_fc_layers) if shared_glu_fc_layers else 0) + n_dependent_glus
        if n_glu_layers <= 0:
            raise ValueError("Invalid Argument: Total number of GLU layers in the feature transformer"
                             " should be greater than 0.")

        self.units = units
        self.norm_factor = tf.math.sqrt(tf.constant(0.5))

        self.glu_layers = list()
        for i in range(n_glu_layers):
            fc_layer = None
            if shared_glu_fc_layers:
                if i < len(shared_glu_fc_layers):
                    fc_layer = shared_glu_fc_layers[i]

            glu_layer = GLULayer(
                units=self.units,
                fc_layer=fc_layer,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            self.glu_layers.append(glu_layer)

    def build(self, input_shape: tf.TensorShape):
        if input_shape[-1] != self.units:
            self.omit_first_residual = True
        else:
            self.omit_first_residual = False

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        for i, glu_layer in enumerate(self.glu_layers):
            x = glu_layer(inputs, training=training)
            if self.omit_first_residual and (i == 0):
                inputs = x
            else:
                x = tf.math.multiply(self.norm_factor, tf.math.add(inputs, x))
                inputs = x

        return x


class Split(tf.keras.layers.Layer):
    def __init__(
            self,
            split_dim: int = 8,
            **kwargs
    ):
        """
        Splits the input tensor into two at a specified column dimension.

        Parameters:
        -----------
        split_dim: int
            Column dimension where the input tensor should be split into two. Default (8).
        """
        super(Split, self).__init__(**kwargs)
        self.split_dim = split_dim

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        return inputs[:, :self.split_dim], inputs[:, self.split_dim:]


class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(
            self,
            units: int,
            n_steps: int = 3,
            epsilon: float = 1e-15,
            lambda_sparse: float = 1e-3,
            virtual_batch_size: Optional[int] = None,
            momentum: float = 0.98,
            mask_type: str = "sparsemax",
            **kwargs
    ):
        """
        Creates an attentive transformer that learns masks to select salient features
        for further analysis. This layer propagates the sparsity regularization loss
        to the Keras model/layer that calls this layer.

        Parameters:
        -----------
        units: int
            Number of units in layer. This layer outputs a mask for your data, so the
            number of units should be the same as your data dimension.
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the
            data has more informative features, the number of steps is higher. Large
            values may lead to overfitting. Default (3).
        epsilon: float
            Prevent computing log(0) by adding a small constant log(0+epsilon). Default (1e-15).
        lambda_sparse: float
            Coefficient for the mask sparsity loss. Important parameter to tune. Lower values
            lead to better performance. Default (1e-3).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller
            than and a factor of the overall batch size. Default (None) runs regular batch
            normalization. If an integer value is specified, GBN is run with that virtual
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values
            correspond to larger impact of batch on the rolling statistics computed in
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        mask_type: str
            mask_type ∈ {"softmax", "entmax", "sparsemax"}. Softmax generates a dense mask.
            Entmax (i.e. entmax 1.5) generates a slightly sparser mask. Default(sparsemax)
            generates a highly sparse mask.
            To learn more, refer: https://arxiv.org/abs/1905.05702.
        """
        super(AttentiveTransformer, self).__init__(**kwargs)
        # for computing sparsity regularization loss
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.lambda_sparse = lambda_sparse

        # attentive transformer layers
        self.fc = tf.keras.layers.Dense(units, use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=virtual_batch_size,
                                                     momentum=momentum)

        if mask_type == "sparsemax":
            self.sparse_activation = tfa.activations.sparsemax
        elif mask_type == "entmax":
            self.sparse_activation = entmax15
        elif mask_type == "softmax":
            self.sparse_activation = tf.nn.softmax
        else:
            raise NotImplementedError(
                "Available options for mask_type: {'sparsemax', 'entmax', 'softmax'}"
            )

    def call(self, inputs: tf.Tensor, prior: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.multiply(prior, x)
        x = self.sparse_activation(x, axis=-1)

        # propagate sparsity loss from current mask
        sparsity_reg_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(-x, tf.math.log(x + self.epsilon)),
                axis=-1
            )
        )
        sparsity_reg_loss /= self.n_steps
        self.add_loss(self.lambda_sparse * sparsity_reg_loss)

        return x


class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(
            self,
            decision_dim: int = 8,
            attention_dim: int = 8,
            n_steps: int = 3,
            n_shared_glus: int = 2,
            n_dependent_glus: int = 2,
            relaxation_factor: float = 1.3,
            epsilon: float = 1e-15,
            virtual_batch_size: Optional[int] = None,
            momentum: float = 0.98,
            mask_type: str = "sparsemax",
            lambda_sparse: float = 1e-3,
            **kwargs
    ):
        """
        Creates a TabNet encoder network.

        Parameters:
        -----------
        decision_dim: int
            Dimension of the decision layer. Typically ranges from 8 to 128. Assuming
            decision_dim to be equal to attention_dim is sensible. Large values may lead
            to overfitting. Default (8).
        attention_dim: int
            Dimension of the attention layer. Typically ranges from 8 to 128. Assuming
            attention_dim to be equal to decision_dim is sensible. Large values may lead
            to overfitting. Default (8).
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the
            data has more informative features, the number of steps is higher. Large
            values may lead to overfitting. Default (3).
        n_shared_glus: int
            Number of shared GLU layers within the Feature Transformer. Increasing the
            number of shared layers is an effective strategy to improve predictive performance
            without a significant increase in the number of parameters. Default (2).
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing
            the number of step-dependent layers is an effective strategy to improve predictive
            performance. Default (2).
        relaxation_factor: float
            Relaxation parameter used to compute the prior in the Attentive Transformer
            layers. Typically ranges from 1.0 to 2.0. This is an important hyperparameter
            to tune in TabNets. Default (1.3).
        epsilon: float
            Prevent computing log(0) by adding a small constant log(0+epsilon). Default (1e-15).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller
            than and a factor of the overall batch size. Default (None) runs regular batch
            normalization. If an integer value is specified, GBN is run with that virtual
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values
            correspond to larger impact of batch on the rolling statistics computed in
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        mask_type: str
            mask_type    {"softmax", "entmax", "sparsemax"}. Softmax generates a dense mask.
            Entmax (i.e. entmax 1.5) generates a slightly sparser mask. Default(sparsemax)
            generates a highly sparse mask.
            To learn more, refer: https://arxiv.org/abs/1905.05702.
        lambda_sparse: float
            Coefficient for the mask sparsity loss. Important parameter to tune. Lower values
            lead to better performance. Default (1e-3).
        """
        super(TabNetEncoder, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.decision_dim = decision_dim
        self.n_steps = n_steps
        self.n_shared_glus = n_shared_glus
        self.n_dependent_glus = n_dependent_glus
        self.relaxation_factor = relaxation_factor
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self.lambda_sparse = lambda_sparse

        # plain batch normalization
        self.initial_bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)

        # shared glu layers
        self.glu_dim = decision_dim + attention_dim
        self.shared_glu_fc_layers = list()
        for _ in range(n_shared_glus):
            self.shared_glu_fc_layers.append(tf.keras.layers.Dense(units=self.glu_dim * 2, use_bias=False))

        # initial feature transformer
        self.initial_feature_transformer = FeatureTransformer(
            n_dependent_glus=self.n_dependent_glus,
            shared_glu_fc_layers=self.shared_glu_fc_layers,
            units=self.glu_dim,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            name="FeatureTransformer_Step_0",
        )

        # split layer
        self.split_layer = Split(split_dim=decision_dim)

    def get_config(self):
        config = {
            'decision_dim': self.decision_dim,
            'attention_dim': self.attention_dim,
            'n_steps': self.n_steps,
            'n_shared_glus': self.n_shared_glus,
            'n_dependent_glus': self.n_dependent_glus,
            'relaxation_factor': self.relaxation_factor,
            'epsilon': self.epsilon,
            'virtual_batch_size': self.virtual_batch_size,
            'momentum': self.momentum,
            'mask_type': self.mask_type,
            'lambda_sparse': self.lambda_sparse
        }
        base_config = super(TabNetEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape: tf.TensorShape):
        feature_dim = input_shape[-1]

        # feature and attentive transformers for each step
        self.step_feature_transformers = list()
        self.step_attentive_transformers = list()
        for step in range(self.n_steps):
            feature_transformer = FeatureTransformer(
                n_dependent_glus=self.n_dependent_glus,
                shared_glu_fc_layers=self.shared_glu_fc_layers,
                units=self.glu_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                name=f"FeatureTransformer_Step_{(step + 1)}",
            )
            attentive_transformer = AttentiveTransformer(
                units=feature_dim,
                n_steps=self.n_steps,
                epsilon=self.epsilon,
                lambda_sparse=self.lambda_sparse,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                mask_type=self.mask_type,
                name=f"AttentiveTransformer_Step_{(step + 1)}",
            )
            self.step_feature_transformers.append(
                feature_transformer
            )
            self.step_attentive_transformers.append(
                attentive_transformer

            )

    def call(self, inputs: tf.Tensor, prior: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        step_output_aggregate = tf.zeros((batch_size, self.decision_dim))

        if prior is None:
            prior = tf.ones_like(inputs)

        x = self.initial_bn(inputs, training=training)
        x_proc = self.initial_feature_transformer(x, training=training)
        _, x_a = self.split_layer(x_proc)

        for step in range(self.n_steps):
            # step operations
            mask = self.step_attentive_transformers[step](x_a,
                                                          prior=prior,
                                                          training=training)
            masked_x = tf.multiply(mask, x)
            x_proc = self.step_feature_transformers[step](masked_x,
                                                          training=training)
            x_d, x_a = self.split_layer(x_proc)
            step_output = tf.keras.activations.relu(x_d)

            # for prediction
            step_output_aggregate += step_output

            # update prior
            prior = tf.multiply(self.relaxation_factor - mask, prior)

        return step_output_aggregate

    def calculate_feature_attribution(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, Dict[int, tf.Tensor]]:
        feature_attribution = tf.zeros_like(inputs)
        masks = dict()
        prior = tf.ones_like(inputs)

        x = self.initial_bn(inputs, training=False)
        x_proc = self.initial_feature_transformer(x, training=False)
        _, x_a = self.split_layer(x_proc)

        for step in range(self.n_steps):
            # step operations
            mask = self.step_attentive_transformers[step](x_a,
                                                          prior=prior,
                                                          training=False)
            masked_x = tf.multiply(mask, x)
            x_proc = self.step_feature_transformers[step](masked_x,
                                                          training=False)
            x_d, x_a = self.split_layer(x_proc)
            step_output = tf.keras.activations.relu(x_d)

            # for interpretation
            masks[step] = mask
            step_coefficient = tf.reshape(
                tf.math.reduce_sum(step_output, axis=-1),
                shape=(-1, 1)
            )
            feature_attribution += tf.multiply(step_coefficient, mask)

            # update prior
            prior = tf.multiply(self.relaxation_factor - mask, prior)

        # normalization
        feature_attribution /= tf.reshape(
            tf.reduce_sum(feature_attribution, axis=-1),
            shape=(-1, 1)
        )

        return feature_attribution, masks


class TabNetDecoder(tf.keras.layers.Layer):
    def __init__(
            self,
            reconstruction_dim: int,
            n_steps: int = 3,
            n_shared_glus: int = 1,
            n_dependent_glus: int = 1,
            virtual_batch_size: Optional[int] = None,
            momentum: float = 0.98,
            **kwargs
    ):
        """
        Creates a TabNet decoder network.

        Parameters
        -----------
        reconstruction_dim: int
            Dimension of the decoder network's output layer.
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the
            data has more informative features, the number of steps is higher. Large
            values may lead to overfitting. Default (3).
        n_shared_glus: int
            Number of shared GLU layers within the Feature Transformer. Increasing the
            number of shared layers is an effective strategy to improve predictive performance
            without a significant increase in the number of parameters. Default (2).
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing
            the number of step-dependent layers is an effective strategy to improve predictive
            performance. Default (2).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller
            than and a factor of the overall batch size. Default (None) runs regular batch
            normalization. If an integer value is specified, GBN is run with that virtual
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values
            correspond to larger impact of batch on the rolling statistics computed in
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(TabNetDecoder, self).__init__(**kwargs)
        self.reconstruction_dim = reconstruction_dim
        self.n_steps = n_steps
        self.n_shared_glus = n_shared_glus
        self.n_dependent_glus = n_dependent_glus
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

    def build(self, input_shape: tf.TensorShape):
        latent_dim = input_shape[-1]

        # shared glu layers
        self.shared_glu_fc_layers = list()
        for _ in range(self.n_shared_glus):
            self.shared_glu_fc_layers.append(tf.keras.layers.Dense(units=latent_dim * 2, use_bias=False))

        # feature and attentive transformers for each step
        self.step_feature_transformers = list()
        self.step_fc_layers = list()
        for step in range(self.n_steps):
            feature_transformer = FeatureTransformer(
                n_dependent_glus=self.n_dependent_glus,
                shared_glu_fc_layers=self.shared_glu_fc_layers,
                units=latent_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                name=f"FeatureTransformer_Decoder_Step_{(step + 1)}",
            )
            fc_layer = tf.keras.layers.Dense(
                units=self.reconstruction_dim,
                use_bias=False,
                name=f"FC_Decoder_Step_{(step + 1)}",
            )
            self.step_feature_transformers.append(
                feature_transformer
            )
            self.step_fc_layers.append(
                fc_layer
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_dim = tf.shape(inputs)[-1]
        reconstructed_features = tf.zeros((batch_dim, self.reconstruction_dim))

        for step in range(self.n_steps):
            # step operation
            x = self.step_feature_transformers[step](inputs,
                                                     training=False)
            x = self.step_fc_layers[step](inputs)
            reconstructed_features += x

        return reconstructed_features


class TabularEmbedding(tf.keras.layers.Layer):
    def __init__(
            self,
            feature_names: List[str],
            categorical_str_indices: List[int],
            categorical_dims: List[int],
            categorical_embed_dims: Union[int, List[int]],
            **kwargs
    ):
        """
        Creates an embedding layer for categorical variables. Numerical variables are
        untouched.

        Parameters
        -----------
        feature_names: List[str]
            A list of feature names in order that they are expected to be input into the
        """
        super(TabularEmbedding, self).__init__(**kwargs)