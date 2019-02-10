/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @addtogroup NeuralNetworks
 * @{
 */
/**
 * @file NeuralNetworks.h
 */
#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_MOCK_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_MOCK_H
/******************************************************************
 *
 * IMPORTANT NOTICE:
 *
 *   This file is part of Android's set of stable system headers
 *   exposed by the Android NDK (Native Development Kit).
 *
 *   Third-party source AND binary code relies on the definitions
 *   here to be FROZEN ON ALL UPCOMING PLATFORM RELEASES.
 *
 *   - DO NOT MODIFY ENUMS (EXCEPT IF YOU ADD NEW 32-BIT VALUES)
 *   - DO NOT MODIFY CONSTANTS OR FUNCTIONAL MACROS
 *   - DO NOT CHANGE THE SIGNATURE OF FUNCTIONS IN ANY WAY
 *   - DO NOT CHANGE THE LAYOUT OR SIZE OF STRUCTURES
 */
#include <android/hardware_buffer.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>
__BEGIN_DECLS
/**
 * Operand types.
 *
 * The type of operands that can be added to a model.
 *
 * Although we define many types, most operators accept just a few
 * types. Most used are {@link ANEURALNETWORKS_TENSOR_FLOAT32},
 * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
 * and {@link ANEURALNETWORKS_INT32}.
 *
 * Available since API level 27.
 */
typedef enum {
    /** A 32 bit floating point scalar value. */
    ANEURALNETWORKS_FLOAT32 = 0,
    /** A signed 32 bit integer scalar value. */
    ANEURALNETWORKS_INT32 = 1,
    /** An unsigned 32 bit integer scalar value. */
    ANEURALNETWORKS_UINT32 = 2,
    /** A tensor of 32 bit floating point values. */
    ANEURALNETWORKS_TENSOR_FLOAT32 = 3,
    /** A tensor of 32 bit integer values. */
    ANEURALNETWORKS_TENSOR_INT32 = 4,
    /**
     * A tensor of 8 bit unsigned integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that be used to convert the
     * 8 bit integer to the real value and vice versa. These two numbers are:
     * - scale: a 32 bit floating point value greater than zero.
     * - zeroPoint: a 32 bit integer, in range [0, 255].
     *
     * The formula is:
     *   real_value = (integer_value - zeroPoint) * scale.
     */
    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 5,
#if __ANDROID_API__ >= __ANDROID_API_Q__
    /**
     * An 8 bit boolean scalar value.
     *
     * Values of this operand type are either true or false. A zero value
     * represents false; any other value represents true.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_BOOL = 6,
    /**
     * A tensor of 16 bit signed integers that represent real numbers.
     *
     * Attached to this tensor is a number representing real value scale that is
     * used to convert the 16 bit number to a real value in the following way:
     * realValue = integerValue * scale.
     *
     * scale is a 32 bit floating point with value greater then zero.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TENSOR_QUANT16_SYMM = 7,
    /**
     * A tensor of IEEE 754 16 bit floating point values.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TENSOR_FLOAT16 = 8,
    /**
     * A tensor of 8 bit boolean values.
     *
     * Values of this operand type are either true or false. A zero value
     * represents false; any other value represents true.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TENSOR_BOOL8 = 9,
    /**
     * An IEEE 754 16 bit floating point scalar value.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_FLOAT16 = 10,
    /**
     * A tensor of 8 bit signed integers that represent real numbers.
     *
     * This tensor is associated with additional fields that can
     * be used to convert the 8 bit signed integer to the real value and vice
     * versa. These fields are:
     * - channelDim: a 32 bit unsigned integer indicating channel dimension.
     * - scales: an array of positive 32 bit floating point values.
     * The size of the scales array must be equal to dimensions[channelDim].
     *
     * {@link ANeuralNetworksModel_setOperandSymmPerChannelQuantParams} must be
     * used to set the parameters for an Operand of this type.
     *
     * The channel dimension of this tensor must not be unknown
     * (dimensions[channelDim] != 0).
     *
     * The formula is:
     * realValue[..., C, ...] =
     *     integerValue[..., C, ...] * scales[C]
     * where C is an index in the Channel dimension.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,
    /**
     * A tensor of 16 bit unsigned integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that can be used to convert the
     * 16 bit integer to the real value and vice versa. These two numbers are:
     * - scale: a 32 bit floating point value greater than zero.
     * - zeroPoint: a 32 bit integer, in range [0, 65535].
     *
     * The formula is:
     * real_value = (integer_value - zeroPoint) * scale.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TENSOR_QUANT16_ASYMM = 12,
    /**
     * A tensor of 8 bit signed integers that represent real numbers.
     *
     * Attached to this tensor is a number representing real value scale that is
     * used to convert the 8 bit number to a real value in the following way:
     * realValue = integerValue * scale.
     *
     * scale is a 32 bit floating point with value greater then zero.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM = 13,
#endif  // __ANDROID_API__ >= __ANDROID_API_Q__
} OperandCode;
/**
 * Operation types.
 *
 * The type of operations that can be added to a model.
 *
 * Available since API level 27.
 */
typedef enum {
    // Operations below are available since API level 27.
    /**
     * Adds two tensors, element-wise.
     *
     * Takes two input tensors of identical {@link OperandCode} and compatible
     * dimensions. The output is the sum of both input tensors, optionally
     * modified by an activation function.
     *
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     *
     * The size of the output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its
     * way forward.
     *
     * Example:
     *
     *     input1.dimension = {4, 1, 2}
     *     input2.dimension = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode}, and compatible dimensions
     *      as input0.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     *
     * Outputs:
     * * 0: The sum, a tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_ADD = 0,
    /**
     * Performs a 2-D average pooling operation.
     *
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, i, j, channel] =
     *         sum_{di, dj}(
     *             input[b, strides[1] * i + di, strides[2] * j + dj, channel]
     *         ) / sum(1)
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both explicit padding and implicit padding are supported.
     *
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      width.
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      height.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 10: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *       Set to true to specify NCHW data layout for input0 and output0.
     *       Available since API level 29.
     *
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      {@link PaddingCode} values.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      width.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      height.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 7: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_AVERAGE_POOL_2D = 1,
    /**
     * Concatenates the input tensors along the given dimension.
     *
     * The input tensors must have identical {@link OperandCode} and the same
     * dimensions except the dimension along the concatenation axis.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} (full support since API
     *   level 29, see the input section)
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0 ~ n-1: The list of n input tensors, of shape
     *            [D0, D1, ..., Daxis(i), ..., Dm].
     *            Before API level 29, all input tensors of
     *            {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *            must have the same scale and zeroPoint as the output tensor.
     * * n: An {@link ANEURALNETWORKS_INT32} scalar, specifying the
     *      concatenation axis.
     *
     * Outputs:
     * * 0: The output, a tensor of the same {@link OperandCode} as the input
     *      tensors. The output shape is [D0, D1, ..., sum(Daxis(i)), ..., Dm].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_CONCATENATION = 2,
    /**
     * Performs an 2-D convolution operation.
     *
     * The CONV_2D op sweeps a 2-D filter that can mix channels together over a
     * batch of images, applying the filter to each window of each image of the
     * appropriate size.
     *
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, i, j, channel] =
     *         sum_{di, dj, k} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
     *             filter[channel, di, dj, k]
     *         ) + bias[channel]
     *
     * Supported tensor {@link OperandCode} configurations:
     * * 32 bit Floating point :
     * * * {@link ANEURALNETWORKS_TENSOR_FLOAT32} for input, filter, output, and
     * bias.
     *
     * * Quantized:
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} for input, filter, and
     * output.
     * * * {@link ANEURALNETWORKS_TENSOR_INT32} for bias (with scale set to
     * * * input.scale * filter.scale).
     *
     * Available since API level 29:
     * * Quantized with symetric per channel quantization for the filter:
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} for input, and output.
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} for filter.
     * * * {@link ANEURALNETWORKS_TENSOR_INT32} for bias (scale set to 0.0,
     * * * each value scaling is separate and equal to input.scale *
     * filter.scales[channel]).
     *
     * * 16 bit Floating point:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} for input, filter, output, and
     * bias.
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both explicit padding and implicit padding are supported.
     *
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_in], specifying the
     *      filter. For tensor of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} the channel
     *      dimension (extraParams.channelQuant.channelDim) must be set to 0.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias must be of the same
     *      type. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the bias should be of {@link
     * ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale ==
     * input_scale * filter_scale. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}, the bias should be of
     * {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale
     * of 0. The actual scale of each value 'i' is equal to bias_scale[i] =
     * input_scale * filter_scale[i].
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 10: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     * * 11: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for width. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on width dimension. If this
     * input is set, input 12 (dilation factor for height) must be specified as
     * well. Available since API level 29.
     * * 12: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for height. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on height dimension. If
     * this input is set, input 11 (dilation factor for width) must be specified
     * as well. Available since API level 29.
     *
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_in], specifying the
     *      filter. For tensor of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} the channel
     *      dimension (extraParams.channelQuant.channelDim) must be set to 0.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias must be of the same
     *      type. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the bias should be of {@link
     * ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale ==
     * input_scale * filter_scale. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}, the bias should be of
     * {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale
     * of 0. The actual scale of each value 'i' is equal to bias_scale[i] =
     * input_scale * filter_scale[i].
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      {@link PaddingCode} values.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 7: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     * * 8: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for width. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on width dimension. If this
     * input is set, input 9 (dilation factor for height) must be specified as
     * well. Available since API level 29.
     * * 9: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for height. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on height dimension. If
     * this input is set, input 8 (dilation factor for width) must be specified
     * as well. Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth_out]. For output tensor of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the following condition
     *      must be satisfied: output_scale > input_scale * filter_scale (for
     *      filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} this condition must be
     * true for all filter scales).
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_CONV_2D = 3,
    /**
     * Performs a depthwise 2-D convolution operation.
     *
     * Given an input tensor of shape [batches, height, width, depth_in] and a
     * filter tensor of shape [1, filter_height, filter_width, depth_out]
     * containing depth_out convolutional filters of depth 1, DEPTHWISE_CONV
     * applies a different filter to each input channel (expanding from 1
     * channel to channel_multiplier channels for each), then concatenates the
     * results together.
     *
     * The output has depth_out = depth_in * depth_multiplier channels.
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, i, j, k * channel_multiplier + q] =
     *         sum_{di, dj} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
     *             filter[1, di, dj, k * channel_multiplier + q]
     *         ) + bias[k * channel_multiplier + q]
     *
     * Supported tensor {@link OperandCode} configurations:
     * * 32 bit Floating point :
     * * * {@link ANEURALNETWORKS_TENSOR_FLOAT32} for input, filter, output, and
     * bias.
     *
     * * Quantized:
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} for input, filter, and
     * output.
     * * * {@link ANEURALNETWORKS_TENSOR_INT32} for bias (with scale set to
     * * * input.scale * filter.scale).
     *
     * Available since API level 29:
     * * Quantized with symetric per channel quantization for the filter:
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} for input, and output.
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} for filter.
     * * * {@link ANEURALNETWORKS_TENSOR_INT32} for bias (scale set to 0.0,
     * * * each value scaling is separate and equal to input.scale *
     * filter.scales[channel]).
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both explicit padding and implicit padding are supported.
     *
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
     *      specifying the filter. For tensor of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} the channel
     *      dimension (extraParams.channelQuant.channelDim) must be set to 3.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias must be of the same
     *      type. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the bias should be of {@link
     * ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale ==
     * input_scale * filter_scale. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}, the bias should be of
     * {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale
     * of 0. The actual scale of each value 'i' is equal to bias_scale[i] =
     * input_scale * filter_scale[i].
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, specifying the depthwise
     *      multiplier.
     * * 10: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *       {@link FuseCode} values. Specifies the activation to
     *       invoke on the result.
     * * 11: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *       Set to true to specify NCHW data layout for input0 and output0.
     *       Available since API level 29.
     * * 12: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for width. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on width dimension. If this
     * input is set, input 13 (dilation factor for height) must be specified as
     * well. Available since API level 29.
     * * 13: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for height. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on height dimension. If
     * this input is set, input 12 (dilation factor for width) must be specified
     * as well. Available since API level 29.
     *
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
     *      specifying the filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias must be of the same
     *      type. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the bias should be of {@link
     * ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale ==
     * input_scale * filter_scale. For filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}, the bias should be of
     * {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and bias_scale
     * of 0. The actual scale of each value 'i' is equal to bias_scale[i] =
     * input_scale * filter_scale[i].
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      {@link PaddingCode} values.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the depthwise
     *      multiplier.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 8: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     * * 9: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for width. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on width dimension. If this
     * input is set, input 10 (dilation factor for height) must be specified as
     * well. Available since API level 29.
     * * 10: An optional {@link ANEURALNETWORKS_INT32} scalar, specifying the
     * dilation factor for height. Defaults to 1. If set to k > 1, there will be
     * k-1 skipped cells between each filter element on height dimension. If
     * this input is set, input 9 (dilation factor for width) must be specified
     * as well. Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth_out]. For output tensor of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the following condition
     *      must be satisfied: output_scale > input_scale * filter_scale (for
     *      filter tensor of {@link
     * ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} this condition must be
     * true for all filter scales).
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_DEPTHWISE_CONV_2D = 4,
    /**
     * Rearranges data from depth into blocks of spatial data.
     *
     * More specifically, this op outputs a copy of the input tensor where
     * values from the depth dimension are moved in spatial blocks to the height
     * and width dimensions. The value block_size indicates the input block size
     * and how the data is moved.
     *
     * Chunks of data of size block_size * block_size from depth are rearranged
     * into non-overlapping blocks of size block_size x block_size.
     *
     * The width of the output tensor is input_depth * block_size, whereas the
     * height is input_height * block_size. The depth of the input tensor must
     * be divisible by block_size * block_size
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the block_size.
     *      block_size must be >=1 and block_size * block_size must be a divisor
     *      of the input depth.
     * * 2: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batch, height*block_size,
     *      width*block_size, depth/(block_size*block_size)].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_DEPTH_TO_SPACE = 5,
    /**
     * Dequantizes the input tensor.
     *
     * The formula is:
     *
     *     output = (input - zeroPoint) * scale.
     *
     * Supported input tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} (since API level
     * 29)
     *
     * Supported output tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}.
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: A tensor with the same shape as input0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_DEQUANTIZE = 6,
    /**
     * Looks up sub-tensors in the input tensor.
     *
     * This operator takes for input a tensor of values (Values) and
     * a one-dimensional tensor of selection indices (Lookups).
     * The output tensor is the concatenation of sub-tensors of Values as
     * selected by Lookups.
     *
     * Think of Values as being sliced along its first dimension:
     * The entries in Lookups select which slices are concatenated together
     * to create the output tensor.
     *
     * For example, if Values has shape of [40, 200, 300] and
     * Lookups has shape of [3], all three values found in Lookups are
     * expected to be between 0 and 39. The resulting tensor must
     * have shape of [3, 200, 300].
     *
     * If a value in Lookups is out of bounds, the operation must fail
     * and an error must be reported.
     *
     * Supported value tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported value tensor rank: from 2
     *
     * Inputs:
     * * 0: Lookups. A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}.
     *      The values are indices into the first dimension of Values.
     * * 1: Values. An n-D tensor, where n >= 2, from which sub-tensors are
     *      extracted.
     *
     * Output:
     * * 0: A n-D tensor with the same rank and shape as the Values
     *      tensor, except for the first dimension which has the same size
     *      as Lookups' only dimension.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_EMBEDDING_LOOKUP = 7,
    /**
     * Computes element-wise floor() on the input tensor.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor, of the same {@link OperandCode} and dimensions as
     *      the input tensor.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_FLOOR = 8,
    /**
     * Denotes a fully (densely) connected layer, which connects all elements
     * in the input tensor with each element in the output tensor.
     *
     * This layer implements the operation:
     *
     *     outputs = activation(inputs * weights’ + bias)
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor of at least rank 2, specifying the input. If rank is
     *      greater than 2, then it gets flattened to a 2-D Tensor. The
     *      (flattened) 2-D Tensor is reshaped (if necessary) to
     *      [batch_size, input_size], where "input_size" corresponds to the
     *      number of inputs to the layer, matching the second dimension of
     *      weights, and "batch_size" is calculated by dividing the number of
     *      elements by "input_size".
     * * 1: A 2-D tensor, specifying the weights, of shape
     *      [num_units, input_size], where "num_units" corresponds to the number
     *      of output nodes.
     * * 2: A 1-D tensor, of shape [num_units], specifying the bias. For input
     *      tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32}, the bias should
     *      also be of {@link ANEURALNETWORKS_TENSOR_FLOAT32}. For input tensor
     *      of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     *
     * Outputs:
     * * 0: The output tensor, of shape [batch_size, num_units]. For output
     *      tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the following
     *      condition must be satisfied:
     *      output_scale > input_scale * filter_scale.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_FULLY_CONNECTED = 9,
    /**
     * Looks up sub-tensors in the input tensor using a key-value map.
     *
     * This operator takes for input a tensor of values (Values),
     * a one-dimensional tensor of selection values (Lookups) and
     * a one-dimensional tensor that maps these values to Values
     * indexes. The output tensor is the concatenation of sub-tensors of
     * Values as selected by Lookups via Keys.
     *
     * Think of Values as being sliced along its outer-most dimension.
     * The output is a concatenation of selected slices, with one slice
     * for each entry of Lookups. The slice selected is the one at the
     * same index as the Maps entry that matches the value in Lookups.
     *
     * For a hit, the corresponding sub-tensor of Values is included
     * in the Output tensor. For a miss, the corresponding sub-tensor in
     * Output must have zero values.
     *
     * For example, if Values has shape of [40, 200, 300],
     * Keys should have a shape of [40]. If Lookups tensor has shape
     * of [3], three slices are being concatenated, so the resulting tensor
     * must have the shape of [3, 200, 300]. If the first entry in Lookups
     * has the value 123456, that value must be located in Keys tensor.
     * If the sixth entry of Keys contains 123456, the sixth slice of Values
     * must be selected. If no entry in Keys has 123456, a slice of zeroes
     * must be concatenated.
     *
     * Supported value tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported value tensor rank: from 2
     *
     * Inputs:
     * * 0: Lookups. A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor with
     *      shape [ k ].
     * * 1: Keys. A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor with shape
     *      [ n ]; Keys and Values pair represent a map, i.e., the ith element
     *      in Keys (Keys[i]) is the key to select the ith sub-tensor in Values
     *      (Values[i]), where 0 <= i <= n-1. Keys tensor *MUST* be sorted in
     *      ascending order.
     * * 2: Values. A tensor with shape of [ n, … ]; i.e., the first dimension
     *      must be n.
     *
     * Outputs:
     * * 0: Output. A tensor with shape [ k …].
     * * 1: Hits. A boolean tensor with shape [ k ] indicates whether the lookup
     *      hits (True) or not (False).
     *      Stored as {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} with offset 0
     *      and scale 1.0f.
     *      A non-zero byte represents True, a hit. A zero indicates otherwise.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_HASHTABLE_LOOKUP = 10,
    /**
     * Applies L2 normalization along the depth dimension.
     *
     * The values in the output tensor are computed as:
     *
     *     output[batch, row, col, channel] =
     *         input[batch, row, col, channel] /
     *         sqrt(sum_{c} pow(input[batch, row, col, c], 2))
     *
     * For input tensor with rank less than 4, independently normalizes each
     * 1-D slice along dimension dim.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     * Tensors with rank less than 4 are only supported since API level 29.
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be normalized.
     * * 1: An optional {@link ANEURALNETWORKS_INT32} scalar, default to -1,
     *      specifying the dimension normalization would be performed on.
     *      Negative index is used to specify axis from the end (e.g. -1 for
     *      the last axis). Must be in the range [-n, n).
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} and same shape as input0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_L2_NORMALIZATION = 11,
    /**
     * Performs an 2-D L2 pooling operation.
     *
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, i, j, c] =
     *         sqrt(sum_{di, dj} pow(input[b, strides[1] * i + di, strides[2] *
     * j + dj, c], 2) / sum(1))
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both explicit padding and implicit padding are supported.
     *
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      width.
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      height.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 10: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *       Set to true to specify NCHW data layout for input0 and output0.
     *       Available since API level 29.
     *
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      {@link PaddingCode} values.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      width.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      height.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 7: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_L2_POOL_2D = 12,
    /**
     * Applies Local Response Normalization along the depth dimension.
     *
     * The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the
     * last dimension), and each vector is normalized independently. Within a
     * given vector, each component is divided by the weighted, squared sum of
     * inputs within depth_radius.
     *
     * The output is calculated using this formula:
     *
     *     sqr_sum[a, b, c, d] = sum(
     *         pow(input[a, b, c, d - depth_radius : d + depth_radius + 1], 2))
     *     output = input / pow((bias + alpha * sqr_sum), beta)
     *
     * For input tensor with rank less than 4, independently normalizes each
     * 1-D slice along specified dimension.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     * Tensors with rank less than 4 are only supported since API level 29.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the radius of
     *      the normalization window.
     * * 2: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the bias, must
     *      not be zero.
     * * 3: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the scale
     *      factor, alpha.
     * * 4: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the exponent,
     *      beta.
     * * 5: An optional {@link ANEURALNETWORKS_INT32} scalar, default to -1,
     *      specifying the dimension normalization would be performed on.
     *      Negative index is used to specify axis from the end (e.g. -1 for
     *      the last axis). Must be in the range [-n, n).
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION = 13,
    /**
     * Computes sigmoid activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = 1 / (1 + exp(-input))
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *      For {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the scale must be 1.f / 256 and the zeroPoint must be 0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_LOGISTIC = 14,
    /**
     * Projects an input to a bit vector via locality senstive hashing.
     *
     * Supported input tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported input tensor rank: from 1
     *
     * Inputs:
     * * 0: Hash functions. Dim.size == 2, DataType: Float.
     *      Tensor[0].Dim[0]: Number of hash functions.
     *      Tensor[0].Dim[1]: Number of projected output bits generated by each
     *      hash function.
     *      If the projection type is Sparse:
     *      Tensor[0].Dim[1] + ceil(log2(Tensor[0].Dim[0])) <= 32
     *
     * * 1: Input. Dim.size >= 1, no restriction on DataType.
     * * 2: Weight. Optional. Dim.size == 1, DataType: Float.
     *      If not set, each input element is considered to have the same weight
     *      of 1.0.
     *      Tensor[1].Dim[0] == Tensor[2].Dim[0]
     * * 3: Type:
     *        Sparse:
     *          Value LSHProjectionType_SPARSE(=3) (since API level 29).
     *          Computed bit vector is considered to be sparse.
     *          Each output element is an int32 made up of multiple bits
     *          computed from hash functions.
     *
     *          NOTE: To avoid collisions across hash functions, an offset value
     *          of k * (1 << Tensor[0].Dim[1]) will be added to each signature,
     *          where k is the index of the hash function.
     *
     *          Value LSHProjectionType_SPARSE_DEPRECATED(=1).
     *          Legacy behavior that does not include the offset value.
     *
     *        Dense:
     *          Value LSHProjectionType_DENSE(=2).
     *          Computed bit vector is considered to be dense. Each output
     *          element represents a bit and can take the value of either
     *          0 or 1.
     *
     * Outputs:
     * * 0: If the projection type is Sparse:
     *      Output.Dim == { Tensor[0].Dim[0] }
     *      A tensor of int32 that represents hash signatures,
     *
     *      If the projection type is Dense:
     *      Output.Dim == { Tensor[0].Dim[0] * Tensor[0].Dim[1] }
     *      A flattened tensor that represents projected bit vectors.
     *
     * Available since API level 27.
     * The offset value for sparse projections was added in API level 29.
     */
    ANEURALNETWORKS_LSH_PROJECTION = 15,
    /**
     * Performs a single time step in a Long Short-Term Memory (LSTM) layer
     *
     * The LSTM operation is described by the following equations.
     *
     * \f{eqnarray*}{
     * i_t =& \sigma(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}+b_i) & \\
     * f_t =& \sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}+b_f) & \\
     * C_t =& clip(f_t \odot C_{t-1} + i_t \odot
     *        g(W_{xc}x_t+W_{hc}h_{t-1}+b_c),\ t_{cell}) & \\
     * o_t =& \sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o) & \\
     *      & & \\
     *      & clip(W_{proj}(o_t \odot g(C_t))+b_{proj},\ t_{proj})
     *      & if\ there\ is\ a\ projection; \\
     * h_t =& & \\
     *      & o_t \odot g(C_t) & otherwise. \\
     * \f}
     * Where:
     * * \f$x_t\f$ is the input,
     * * \f$i_t\f$ is the input gate,
     * * \f$f_t\f$ is the forget gate,
     * * \f$C_t\f$ is the cell state,
     * * \f$o_t\f$ is the output,
     * * \f$h_t\f$ is the output state,
     * * \f$\sigma\f$ is the logistic sigmoid function,
     * * \f$g\f$ is the cell input and cell output activation function, usually
     *   \f$tahn\f$,
     * * \f$W_{xi}\f$ is the input-to-input weight matrix,
     * * \f$W_{hi}\f$ is the recurrent to input weight matrix,
     * * \f$W_{ci}\f$ is the cell-to-input weight matrix,
     * * \f$b_i\f$ is the input gate bias,
     * * \f$W_{xf}\f$ is the input-to-forget weight matrix,
     * * \f$W_{hf}\f$ is the recurrent-to-forget weight matrix,
     * * \f$W_{cf}\f$ is the cell-to-forget weight matrix,
     * * \f$b_f\f$ is the forget gate bias,
     * * \f$W_{xc}\f$ is the input-to-cell weight matrix,
     * * \f$W_{hc}\f$ is the recurrent-to-cell weight matrix,
     * * \f$b_c\f$ is the cell bias,
     * * \f$W_{xo}\f$ is the input-to-output weight matrix,
     * * \f$W_{ho}\f$ is the recurrent-to-output weight matrix,
     * * \f$W_{co}\f$ is the cell-to-output weight matrix,
     * * \f$b_o\f$ is the output gate bias,
     * * \f$W_{proj}\f$ is the projection weight matrix,
     * * \f$b_{proj}\f$ is the projection bias,
     * * \f$t_{cell}\f$ is the threshold for clipping the cell state, and
     * * \f$t_{proj}\f$ is the threshold for clipping the projected output.
     * * \f$\odot\f$ is the
     *   <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)">
     *   Hadamard product</a> that takes two matrices and produces another
     *   matrix, each element of which is the product of the corresponding
     *   elements of the input matrices.
     *
     * Since API level 29 LSTM supports layer normalization.
     * In case layer normalization is used, the inputs to internal activation
     * functions (sigmoid and \f$g\f$) are normalized, rescaled and recentered
     * following an approach from section 3.1 from
     * https://arxiv.org/pdf/1607.06450.pdf
     *
     * The operation has the following independently optional inputs:
     * * The input-to-input weights (\f$W_{xi}\f$), recurrent-to-input weights
     *   (\f$W_{hi}\f$), cell-to-input (\f$W_{ci}\f$) weights, and input gate
     *   bias (\f$b_i\f$) either all have values, or none of them have values
     *   (i.e., all set to null). If they have no values, coupling of input and
     *   forget gates (CIFG) is used, in which case the input gate (\f$i_t\f$)
     *   is calculated using the following equation instead.
     *   \f{eqnarray*}{
     *   i_t = 1 - f_t
     *   \f}
     * * The cell-to-forget weights (\f$W_{cf}\f$) and cell-to-output weights
     *   (\f$W_{co}\f$) either both have values or neither of them have values.
     *   If they have values, the peephole optimization is used. Additionally,
     *   if CIFG is not used, cell-to-input weights (\f$W_{ci}\f$) is also
     *   required to have values for peephole optimization.
     * * The projection weights (\f$W_{proj}\f$) is required only for the
     *   recurrent projection layer, and should otherwise have no value.
     * * The projection bias (\f$b_{proj}\f$) may (but not required to) have a
     *   value if the recurrent projection layer exists, and should otherwise
     *   have no value.
     * * (API level >= 29) The four layer normalization weights either all have
     *   values or none of them have values. Layer normalization is used when
     *   values are present.
     *
     * References:
     *
     * The default non-peephole non-CIFG implementation is based on:
     * http://www.bioinf.jku.at/publications/older/2604.pdf
     * S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural
     * Computation, 9(8):1735-1780, 1997.
     *
     * The peephole implementation and projection layer is based on:
     * https://research.google.com/pubs/archive/43905.pdf
     * Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory
     * recurrent neural network architectures for large scale acoustic
     * modeling." INTERSPEECH, 2014.
     * (However, the concept of peephole optimization was introduced in work
     * prior to this paper.)
     *
     * The coupling of input and forget gate (CIFG) is based on:
     * http://arxiv.org/pdf/1503.04069.pdf
     * Greff et al. "LSTM: A Search Space Odyssey"
     *
     * The layer normalization is based on:
     * https://arxiv.org/pdf/1607.06450.pdf
     * Jimmy Ba et al. "Layer Normalization"
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * All input and output tensors must be of the same type.
     *
     * Inputs:
     * * 0: The input (\f$x_t\f$).
     *      A 2-D tensor of shape [batch_size, input_size], where “batch_size”
     *      corresponds to the batching dimension, and “input_size” is the size
     *      of the input.
     * * 1: The input-to-input weights (\f$W_{xi}\f$). Optional.
     *      A 2-D tensor of shape [num_units, input_size], where “num_units”
     *      corresponds to the number of cell units.
     * * 2: The input-to-forget weights (\f$W_{xf}\f$).
     *      A 2-D tensor of shape [num_units, input_size].
     * * 3: The input-to-cell weights (\f$W_{xc}\f$).
     *      A 2-D tensor of shape [num_units, input_size].
     * * 4: The input-to-output weights (\f$W_{xo}\f$).
     *      A 2-D tensor of shape [num_units, input_size].
     * * 5: The recurrent-to-input weights (\f$W_{hi}\f$). Optional.
     *      A 2-D tensor of shape [num_units, output_size], where “output_size”
     *      corresponds to either the number of cell units (i.e., “num_units”),
     *      or the second dimension of the “projection_weights”, if defined.
     * * 6: The recurrent-to-forget weights (\f$W_{hf}\f$).
     *      A 2-D tensor of shape [num_units, output_size].
     * * 7: The recurrent-to-cell weights (\f$W_{hc}\f$).
     *      A 2-D tensor of shape [num_units, output_size].
     * * 8: The recurrent-to-output weights (\f$W_{ho}\f$).
     *      A 2-D tensor of shape [num_units, output_size].
     * * 9: The cell-to-input weights (\f$W_{ci}\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 10:The cell-to-forget weights (\f$W_{cf}\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 11:The cell-to-output weights (\f$W_{co}\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 12:The input gate bias (\f$b_i\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 13:The forget gate bias (\f$b_f\f$).
     *      A 1-D tensor of shape [num_units].
     * * 14:The cell bias (\f$b_c\f$).
     *      A 1-D tensor of shape [num_units].
     * * 15:The output gate bias (\f$b_o\f$).
     *      A 1-D tensor of shape [num_units].
     * * 16:The projection weights (\f$W_{proj}\f$). Optional.
     *      A 2-D tensor of shape [output_size, num_units].
     * * 17:The projection bias (\f$b_{proj}\f$). Optional.
     *      A 1-D tensor of shape [output_size].
     * * 18:The output state (in) (\f$h_{t-1}\f$).
     *      A 2-D tensor of shape [batch_size, output_size].
     * * 19:The cell state (in) (\f$C_{t-1}\f$).
     *      A 2-D tensor of shape [batch_size, num_units].
     * * 20:The activation function (\f$g\f$).
     *      A value indicating the activation function:
     *      <ul>
     *      <li>0: None;
     *      <li>1: Relu;
     *      <li>3: Relu6;
     *      <li>4: Tanh;
     *      <li>6: Sigmoid.
     *      </ul>
     * * 21:The clipping threshold (\f$t_{cell}\f$) for the cell state, such
     *      that values are bound within [-cell_clip, cell_clip]. If set to 0.0
     *      then clipping is disabled.
     *      Until API level 29 this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT32}. Since API level 29, if all the input
     *      tensors have type {@link ANEURALNETWORKS_TENSOR_FLOAT32}, this
     *      scalar must be of the type {@link ANEURALNETWORKS_FLOAT32},
     *      otherwise if all the input tensors have the type {@link
     *      ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT16}.
     * * 22:The clipping threshold (\f$t_{proj}\f$) for the output from the
     *      projection layer, such that values are bound within
     *      [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     *      Until API level 29 this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT32}. Since API level 29, if all the input
     *      tensors have type {@link ANEURALNETWORKS_TENSOR_FLOAT32}, this
     *      scalar must be of the type {@link ANEURALNETWORKS_FLOAT32},
     *      otherwise if all the input tensors have the type {@link
     *      ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT16}.
     * Since API level 29 there are additional inputs to this op:
     * * 23:The input layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at input gate.
     * * 24:The forget layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at forget gate.
     * * 25:The cell layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at cell gate.
     * * 26:The output layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at output gate.
     *
     * Outputs:
     * * 0: The scratch buffer.
     *      A 2-D tensor of shape [batch_size, num_units * 3] with CIFG, or
     *      [batch_size, num_units * 4] without CIFG.
     * * 1: The output state (out) (\f$h_t\f$).
     *      A 2-D tensor of shape [batch_size, output_size].
     * * 2: The cell state (out) (\f$C_t\f$).
     *      A 2-D tensor of shape [batch_size, num_units].
     * * 3: The output (\f$o_t\f$).
     *      A 2-D tensor of shape [batch_size, output_size]. This is effectively
     *      the same as the current “output state (out)” value.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_LSTM = 16,
    /**
     * Performs an 2-D max pooling operation.
     *
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, i, j, channel] =
     *         max_{di, dj} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj, channel]
     *         )
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both explicit padding and implicit padding are supported.
     *
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      width.
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      height.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 10: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *       Set to true to specify NCHW data layout for input0 and output0.
     *       Available since API level 29.
     *
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      {@link PaddingCode} values.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      width.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the filter
     *      height.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 7: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_MAX_POOL_2D = 17,
    /**
     * Multiplies two tensors, element-wise.
     *
     * Takes two input tensors of identical {@link OperandCode} and compatible
     * dimensions. The output is the product of both input tensors, optionally
     * modified by an activation function.
     *
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     *
     * The size of the resulting output is the maximum size along each dimension
     * of the input operands. It starts with the trailing dimensions, and works
     * its way forward.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode}, and compatible dimensions
     *      as input0.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     *
     * Outputs:
     * * 0: The product, a tensor of the same {@link OperandCode} as input0.
     *      For output tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the following condition must be satisfied:
     *      output_scale > input1_scale * input2_scale.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_MUL = 18,
    /**
     * Computes rectified linear activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = max(0, input)
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_RELU = 19,
    /**
     * Computes rectified linear 1 activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = min(1.f, max(-1.f, input))
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_RELU1 = 20,
    /**
     * Computes rectified linear 6 activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = min(6, max(0, input))
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_RELU6 = 21,
    /**
     * Reshapes a tensor.
     *
     * Given tensor, this operation returns a tensor that has the same values as
     * tensor, but with a newly specified shape.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the tensor to be reshaped.
     * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, defining the
     *      shape of the output tensor. The number of elements implied by shape
     *      must be the same as the number of elements in the input tensor.
     *
     * Outputs:
     * * 0: The output tensor, of shape specified by the input shape.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_RESHAPE = 22,
    /**
     * Resizes images to given size using the bilinear interpretation.
     *
     * Resized images must be distorted if their output aspect ratio is not the
     * same as input aspect ratio. The corner pixels of output may not be the
     * same as corner pixels of input.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      height of the output tensor.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      width of the output tensor.
     * * 3: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, new_height, new_width, depth].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_RESIZE_BILINEAR = 23,
    /**
     * A basic recurrent neural network layer.
     *
     * This layer implements the operation:
     * outputs = state = activation(inputs * input_weights +
     *                              state * recurrent_weights + bias)
     *
     * Where:
     * * “input_weights” is a weight matrix that multiplies the inputs;
     * * “recurrent_weights” is a weight matrix that multiplies the current
     *    “state” which itself is the output from the previous time step
     *    computation;
     * * “bias” is a bias vector (added to each output vector in the batch);
     * * “activation” is the function passed as the “fused_activation_function”
     *   argument (if not “NONE”).
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * The input tensors must all be the same type.
     *
     * Inputs:
     * * 0: input.
     *      A 2-D tensor of shape [batch_size, input_size], where “batch_size”
     *      corresponds to the batching dimension, and “input_size” is the size
     *      of the input.
     * * 1: weights.
     *      A 2-D tensor of shape [num_units, input_size], where “num_units”
     *      corresponds to the number of units.
     * * 2: recurrent_weights.
     *      A 2-D tensor of shape [num_units, num_units], with columns
     *      corresponding to the weights from each unit.
     * * 3: bias.
     *      A 1-D tensor of shape [num_units].
     * * 4: hidden state (in).
     *      A 2-D tensor of shape [batch_size, num_units].
     * * 5: fused_activation_function.
     *      An optional {@link FuseCode} value indicating the
     *      activation function. If “NONE” is specified then it results in a
     *      linear activation.
     *
     * Outputs:
     * * 0: hidden state (out).
     *      A 2-D tensor of shape [batch_size, num_units].
     *
     * * 1: output.
     *      A 2-D tensor of shape [batch_size, num_units]. This is effectively
     *      the same as the current state value.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_RNN = 24,
    /**
     * Computes the softmax activation on the input tensor element-wise, per
     * batch, by normalizing the input vector so the maximum coefficient is
     * zero.
     *
     * The output is calculated using this formula:
     *
     *     output[batch, i] =
     *         exp((input[batch, i] - max(input[batch, :])) * beta) /
     *         sum_{k}{exp((input[batch, k] - max(input[batch, :])) * beta)}
     *
     * For input tensor with rank other than 2, the activation will be applied
     * independently on each 1-D slice along specified dimension.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     * Tensors with rank other than 2 or 4 are only supported since API
     * level 29.
     *
     * Inputs:
     * * 0: A 2-D or 4-D tensor, specifying the tensor to be reshaped.
     * * 1: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the positive
     *      scaling factor for the exponent, beta.
     * * 2: An optional {@link ANEURALNETWORKS_INT32} scalar, default to -1,
     *      specifying the dimension the activation would be performed on.
     *      Negative index is used to specify axis from the end (e.g. -1 for
     *      the last axis). Must be in the range [-n, n).
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *      For {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the scale must be 1.f / 256 and the zeroPoint must be 0.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_SOFTMAX = 25,
    /**
     * Rearranges blocks of spatial data, into depth.
     *
     * More specifically, this op outputs a copy of the input tensor where
     * values from the height and width dimensions are moved to the depth
     * dimension. The value block_size indicates the input block size and how
     * the data is moved.
     *
     * Chunks of data of size block_size * block_size from depth are rearranged
     * into non-overlapping blocks of size block_size x block_size.
     *
     * The depth of the output tensor is input_depth * block_size * block_size.
     * The input tensor's height and width must be divisible by block_size.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the block_size.
     *      block_size must be >=1 and block_size must be a divisor of both the
     *      input height and width.
     * * 2: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, height/block_size,
     *      width/block_size, depth_in*block_size*block_size].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_SPACE_TO_DEPTH = 26,
    /**
     * SVDF op is a kind of stateful layer derived from the notion that a
     * densely connected layer that's processing a sequence of input frames can
     * be approximated by using a singular value decomposition of each of its
     * nodes. The implementation is based on:
     *
     * https://research.google.com/pubs/archive/43813.pdf
     *
     * P. Nakkiran, R. Alvarez, R. Prabhavalkar, C. Parada.
     * “Compressing Deep Neural Networks using a Rank-Constrained Topology”.
     * INTERSPEECH, 2015.
     *
     * It processes the incoming input using a 2-stage filtering mechanism:
     * * stage 1 performs filtering on the "features" dimension, whose outputs
     *   get pushed into a memory of fixed-size memory_size.
     * * stage 2 performs filtering on the "time" dimension of the memory_size
     *   memoized outputs of stage 1.
     *
     * Specifically, for rank 1, this layer implements the operation:
     *
     *     memory = push(conv1d(inputs, weights_feature, feature_dim,
     *                          "ANEURALNETWORKS_PADDING_VALID"));
     *     outputs = activation(memory * weights_time + bias);
     *
     * Where:
     * * “weights_feature” is a weights matrix that processes the inputs (by
     *   convolving the input with every “feature filter”), and whose outputs
     *   get pushed, stacked in order, into the fixed-size “memory” (the oldest
     *   entry gets dropped);
     * * “weights_time” is a weights matrix that processes the “memory” (by a
     *   batched matrix multiplication on the num_units);
     * * “bias” is an optional bias vector (added to each output vector in the
     *   batch); and
     * * “activation” is the function passed as the “fused_activation_function”
     *   argument (if not “NONE”).
     *
     * Each rank adds a dimension to the weights matrices by means of stacking
     * the filters.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * All input tensors must be the same type.
     *
     * Inputs:
     * * 0: input.
     *      A 2-D tensor of shape [batch_size, input_size], where “batch_size”
     *      corresponds to the batching dimension, and “input_size” is the size
     *      of the input.
     * * 1: weights_feature.
     *      A 2-D tensor of shape [num_units, input_size], where “num_units”
     *      corresponds to the number of units.
     * * 2: weights_time.
     *      A 2-D tensor of shape [num_units, memory_size], where “memory_size”
     *      corresponds to the fixed-size of the memory.
     * * 3: bias.
     *      An optional 1-D tensor of shape [num_units].
     * * 4: state (in).
     *      A 2-D tensor of shape [batch_size, (memory_size - 1) * num_units *
     * rank].
     * * 5: rank.
     *      The rank of the SVD approximation.
     * * 6: fused_activation_function.
     *      An optional {@link FuseCode} value indicating the
     *      activation function. If “NONE” is specified then it results in a
     *      linear activation.
     *
     * Outputs:
     * * 0: state (out).
     *      A 2-D tensor of the same {@link OperandCode} as the inputs, with
     * shape [batch_size, (memory_size - 1) * num_units * rank].
     * * 1: output.
     *      A 2-D tensor of the same {@link OperandCode} as the inputs, with
     * shape [batch_size, num_units].
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_SVDF = 27,
    /**
     * Computes hyperbolic tangent of input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = tanh(input)
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} (since API level 29)
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *      For {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the scale must be 1.f / 128 and the zeroPoint must be 128.
     *
     * Available since API level 27.
     */
    ANEURALNETWORKS_TANH = 28,
    // Operations below are available since API level 28.
    // TODO: make the description easier to understand.
    /**
     * BatchToSpace for N-dimensional tensors.
     *
     * This operation reshapes the batch dimension (dimension 0) into M + 1
     * dimensions of shape block_shape + [batch], interleaves these blocks back
     * into the grid defined by the spatial dimensions [1, ..., M], to obtain a
     * result with the same rank as the input.
     *
     * This is the reverse of SpaceToBatch.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be reshaped
     * * 1: A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the block
     *      sizes for each spatial dimension of the input tensor. All values
     *      must be >= 1.
     * * 2: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_BATCH_TO_SPACE_ND = 29,
    /**
     * Element-wise division of two tensors.
     *
     * Takes two input tensors of identical {@link OperandCode} and compatible
     * dimensions. The output is the result of dividing the first input tensor
     * by the second, optionally modified by an activation function.
     *
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     *
     * The size of the output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its way
     * forward.
     *
     * Example:
     *     input1.dimension =    {4, 1, 2}
     *     input2.dimension = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the first input.
     * * 1: A tensor of the same {@link OperandCode}, and compatible dimensions
     *      as input0.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_DIV = 30,
    /**
     * Computes the mean of elements across dimensions of a tensor.
     *
     * Reduces the input tensor along the given dimensions to reduce. Unless
     * keep_dims is true, the rank of the tensor is reduced by 1 for each entry
     * in axis. If keep_dims is true, the reduced dimensions are retained with
     * length 1.
     *
     * If dimensions to reduce have no entries, all dimensions are reduced, and
     * a tensor with a single element is returned.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     * * 1: A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions
     *      to reduce. If None (the default), reduces all dimensions. Must be in
     *      the range [-rank(input_tensor), rank(input_tensor)).
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, keep_dims. If positive,
     *      retains reduced dimensions with length 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_MEAN = 31,
    /**
     * Pads a tensor with zeros.
     *
     * This operation pads a tensor according to the specified paddings.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be padded.
     * * 1: A 2-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the paddings
     *      for each spatial dimension of the input tensor. The shape of the
     *      tensor must be {rank(input0), 2}.
     *      padding[i, 0] specifies the number of elements to be padded in the
     *      front of dimension i.
     *      padding[i, 1] specifies the number of elements to be padded after
     * the end of dimension i.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0. The
     *      output tensor has the same rank as input0, and each
     *      dimension of the output tensor has the same size as the
     *      corresponding dimension of the input tensor plus the size
     *      of the padding:
     *          output0.dimension[i] =
     *              padding[i, 0] + input0.dimension[i] + padding[i, 1]
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_PAD = 32,
    // TODO: make the description easier to understand.
    /**
     * SpaceToBatch for N-Dimensional tensors.
     *
     * This operation divides "spatial" dimensions [1, ..., M] of the input into
     * a grid of blocks of shape block_shape, and interleaves these blocks with
     * the "batch" dimension (0) such that in the output, the spatial dimensions
     * [1, ..., M] correspond to the position within the grid, and the batch
     * dimension combines both the position within a spatial block and the
     * original batch position. Prior to division into blocks, the spatial
     * dimensions of the input are optionally zero padded according to paddings.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the input.
     * * 1: A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the block
     *      sizes for each spatial dimension of the input tensor. All values
     *      must be >= 1.
     * * 2: A 2-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the paddings
     *      for each spatial dimension of the input tensor. All values must be
     *      >= 0. The shape of the tensor must be {rank(input0), 2}.
     *      padding[i, 0] specifies the number of element to be padded in the
     *      front of dimension i.
     *      padding[i, 1] specifies the number of element to be padded after the
     *      end of dimension i.
     * * 3: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_SPACE_TO_BATCH_ND = 33,
    /**
     * Removes dimensions of size 1 from the shape of a tensor.
     *
     * Given a tensor input, this operation returns a tensor of the same
     * {@link OperandCode} with all dimensions of size 1 removed. If you don't
     * want to remove all size 1 dimensions, you can remove specific size 1
     * dimensions by specifying the axes (input1).
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, the tensor to be squeezed.
     * * 1: An optional 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The
     *      dimensions to squeeze. If specified only squeezes the dimensions
     *      listed. Otherwise, squeezes all dimensions. The dimension index
     *      starts at 0. An error must be reported if squeezing a dimension that
     *      is not 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0. Contains the
     *      same data as input, but has one or more dimensions of size 1
     *      removed.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_SQUEEZE = 34,
    /**
     * Extracts a strided slice of a tensor.
     *
     * Roughly speaking, this op extracts a slice of size (end - begin) / stride
     * from the given input tensor. Starting at the location specified by begin
     * the slice continues by adding stride to the index until all dimensions
     * are not less than end. Note that a stride can be negative, which causes a
     * reverse slice.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be sliced.
     * * 1: begin, a 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The
     *      starts of the dimensions of the input tensor to be sliced. The
     *      length must be of rank(input0).
     * * 2: end, a 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The
     *      ends of the dimensions of the input tensor to be sliced. The length
     *      must be of rank(input0).
     * * 3: strides, a 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The
     *      strides of the dimensions of the input tensor to be sliced. The
     *      length must be of rank(input0). The entries must be non-zero.
     * * 4: begin_mask, an {@link ANEURALNETWORKS_INT32} scalar. If the ith bit
     *      of begin_mask is set, begin[i] is ignored and the fullest possible
     *      range in that dimension is used instead.
     * * 5: end_mask, an {@link ANEURALNETWORKS_INT32} scalar. If the ith bit of
     *      end_mask is set, end[i] is ignored and the fullest possible range in
     *      that dimension is used instead.
     * * 6: shrink_axis_mask, an {@link ANEURALNETWORKS_INT32} scalar. If the
     *      ith bit of shrink_axis_mask is set, the ith dimension specification
     *      shrinks the dimensionality by 1, taking on the value at index
     *      begin[i]. In this case, the ith specification must define a
     *      slice of size 1, e.g. begin[i] = x, end[i] = x + 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0 and rank (n - k),
     *      where k is the number of bits set in shrink_axis_mask.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_STRIDED_SLICE = 35,
    /**
     * Element-wise subtraction of two tensors.
     *
     * Takes two input tensors of identical {@link OperandCode} and compatible
     * dimensions. The output is the result of subtracting the second input
     * tensor from the first one, optionally modified by an activation function.
     *
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     *
     * The size of the output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its way
     * forward.
     *
     * Example:
     *     input1.dimension =    {4, 1, 2}
     *     input2.dimension = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} (since API level 29)
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the first input.
     * * 1: A tensor of the same {@link OperandCode}, and compatible dimensions
     *      as input0.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_SUB = 36,
    /**
     * Transposes the input tensor, permuting the dimensions according to the
     * perm tensor.
     *
     * The returned tensor's dimension i corresponds to the input dimension
     * perm[i]. If perm is not given, it is set to (n-1...0), where n is the
     * rank of the input tensor. Hence by default, this operation performs a
     * regular matrix transpose on 2-D input Tensors.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be transposed.
     * * 1: An optional 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32},
     *      the permutation of the dimensions of the input tensor.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 28.
     */
    ANEURALNETWORKS_TRANSPOSE = 37,
    // Operations below are available since API level 29.
    /**
     * Computes the absolute value of a tensor, element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_ABS = 38,
    /**
     * Returns the index of the largest element along an axis.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: An n-D tensor specifying the input. Must be non-empty.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar specifying the axis to
     *      reduce across. Negative index is used to specify axis from the
     *      end (e.g. -1 for the last axis). Must be in the range [-n, n).
     *
     * Outputs:
     * * 0: An (n - 1)-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor.
     *
     * Available since API level 29.
     */
    // There is no underscore in ARG_MAX to avoid name conflict with
    // the macro defined in libc/kernel/uapi/linux/limits.h.
    ANEURALNETWORKS_ARGMAX = 39,
    /**
     * Returns the index of the smallest element along an axis.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: An n-D tensor specifying the input. Must be non-empty.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar specifying the axis to
     *      reduce across. Negative index is used to specify axis from the
     *      end (e.g. -1 for the last axis). Must be in the range [-n, n).
     *
     * Outputs:
     * * 0: An (n - 1)-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_ARGMIN = 40,  // See ARGMAX for naming discussion.
    /**
     * Transform axis-aligned bounding box proposals using bounding box deltas.
     *
     * Given the positions of bounding box proposals and the corresponding
     * bounding box deltas for each class, return the refined bounding box
     * regions. The resulting bounding boxes are cliped against the edges of
     * the image.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Inputs:
     * * 0: A 2-D Tensor of shape [num_rois, 4], specifying the locations of the
     *      bounding box proposals, each line with format [x1, y1, x2, y2].
     *      For tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM},
     *      the zeroPoint must be 0 and the scale must be 0.125.
     * * 1: A 2-D Tensor of shape [num_rois, num_classes * 4], specifying the
     *      bounding box delta for each region of interest and each class. The
     *      bounding box deltas are organized in the following order
     *      [dx, dy, dw, dh], where dx and dy is the relative correction factor
     *      for the center position of the bounding box with respect to the
     * width and height, dw and dh is the log-scale relative correction factor
     *      for the width and height. For input0 of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, this tensor should be
     *      of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}.
     * * 2: An 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     *      [batches], specifying the number of output boxes for each batch.
     * * 3: A 2-D Tensor of shape [batches, 2], specifying the information of
     *      each image in the batch, each line with format
     *      [image_height, image_width].
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0, with shape
     *      [num_rois, num_classes * 4], specifying the coordinates of each
     *      output bounding box for each class, with format [x1, y1, x2, y2].
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_AXIS_ALIGNED_BBOX_TRANSFORM = 41,
    /**
     * Performs a forward LSTM on the input followed by a backward LSTM.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 3, either time-major or batch-major.
     *
     * All input and output tensors must be of the same type.
     *
     *
     * Inputs:
     * * 0: The input.
     *      A 3-D tensor of shape:
     *        If time-major: [max_time, batch_size, output_size]
     *        If batch-major: [batch_size, max_time, output_size]
     *      where "max_time" is the number of timesteps (sequence length),
     *      "batch_size" corresponds to the batching dimension, and
     *      "input_size" is the size of the input.
     * * 1: The forward input-to-input weights. Optional.
     *      A 2-D tensor of shape [num_units, input_size], where “num_units”
     *      corresponds to the number of cell units.
     * * 2: The forward input-to-forget weights.
     *      A 2-D tensor of shape [num_units, input_size].
     * * 3: The forward input-to-cell weights.
     *      A 2-D tensor of shape [num_units, input_size].
     * * 4: The forward input-to-output weights.
     *      A 2-D tensor of shape [num_units, input_size].
     * * 5: The forward recurrent-to-input weights. Optional.
     *      A 2-D tensor of shape [num_units, output_size], where “output_size”
     *      corresponds to either the number of cell units (i.e., “num_units”),
     *      or the second dimension of the “projection_weights”, if defined.
     * * 6: The forward recurrent-to-forget weights.
     *      A 2-D tensor of shape [num_units, output_size].
     * * 7: The forward recurrent-to-cell weights.
     *      A 2-D tensor of shape [num_units, output_size].
     * * 8: The forward recurrent-to-output weights.
     *      A 2-D tensor of shape [num_units, output_size].
     * * 9: The forward cell-to-input weights. Optional.
     *      A 1-D tensor of shape [num_units].
     * * 10: The forward cell-to-forget weights. Optional.
     *       A 1-D tensor of shape [num_units].
     * * 11: The forward cell-to-output weights. Optional.
     *       A 1-D tensor of shape [num_units].
     * * 12: The forward input gate bias. Optional.
     *       A 1-D tensor of shape [num_units].
     * * 13: The forward forget gate bias.
     *       A 1-D tensor of shape [num_units].
     * * 14: The forward cell gate bias.
     *       A 1-D tensor of shape [num_units].
     * * 15: The forward output gate bias.
     *       A 1-D tensor of shape [num_units].
     * * 16: The forward projection weights. Optional.
     *       A 2-D tensor of shape [output_size, num_units].
     * * 17: The forward projection bias. Optional.
     *       A 1-D tensor of shape [output_size].
     * * 18: The backward input-to-input weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size], where “num_units”
     *       corresponds to the number of cell units.
     * * 19: The backward input-to-forget weights.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 20: The backward input-to-cell weights.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 21: The backward input-to-output weights.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 22: The backward recurrent-to-input weights. Optional.
     *       A 2-D tensor of shape [num_units, output_size], where “output_size”
     *       corresponds to either the number of cell units (i.e., “num_units”),
     *       or the second dimension of the “projection_weights”, if defined.
     * * 23: The backward recurrent-to-forget weights.
     *       A 2-D tensor of shape [num_units, output_size].
     * * 24: The backward recurrent-to-cell weights.
     *       A 2-D tensor of shape [num_units, output_size].
     * * 25: The backward recurrent-to-output weights.
     *       A 2-D tensor of shape [num_units, output_size].
     * * 26: The backward cell-to-input weights. Optional.
     *       A 1-D tensor of shape [num_units].
     * * 27: The backward cell-to-forget weights. Optional.
     *       A 1-D tensor of shape [num_units].
     * * 28: The backward cell-to-output weights. Optional.
     *       A 1-D tensor of shape [num_units].
     * * 29: The backward input gate bias. Optional.
     *       A 1-D tensor of shape [num_units].
     * * 30: The backward forget gate bias.
     *       A 1-D tensor of shape [num_units].
     * * 31: The backward cell gate bias.
     *       A 1-D tensor of shape [num_units].
     * * 32: The backward output gate bias.
     *       A 1-D tensor of shape [num_units].
     * * 33: The backward projection weights. Optional.
     *       A 2-D tensor of shape [output_size, num_units].
     * * 34: The backward projection bias. Optional.
     *       A 1-D tensor of shape [output_size].
     * * 35: The forward input activation state.
     *       A 2-D tensor of shape [batch_size, output_size].
     * * 36: The forward input cell state.
     *       A 2-D tensor of shape [batch_size, num_units].
     * * 37: The backward input activation state.
     *       A 2-D tensor of shape [batch_size, output_size].
     * * 38: The backward input cell state.
     *       A 2-D tensor of shape [batch_size, num_units].
     * * 39: The auxiliary input. Optional.
     *       A 3-D tensor of shape [max_time, batch_size, input_size], where
     * “batch_size” corresponds to the batching dimension, and “input_size” is
     * the size of the input.
     * * 40: The forward auxiliary input-to-input weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 41: The forward auxiliary input-to-forget weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 42: The forward auxiliary input-to-cell weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 43: The forward auxiliary input-to-output weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 44: The backward auxiliary input-to-input weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 45: The backward auxiliary input-to-forget weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 46: The backward auxiliary input-to-cell weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 47: The backward auxiliary input-to-output weights. Optional.
     *       A 2-D tensor of shape [num_units, input_size].
     * * 48: The activation function.
     *       A value indicating the activation function:
     *       <ul>
     *       <li>0: None;
     *       <li>1: Relu;
     *       <li>3: Relu6;
     *       <li>4: Tanh;
     *       <li>6: Sigmoid.
     *       </ul>
     * * 49: The clipping threshold for the cell state, such
     *       that values are bound within [-cell_clip, cell_clip]. If set to 0.0
     *       then clipping is disabled.
     *       If all the input tensors have type {@link
     * ANEURALNETWORKS_TENSOR_FLOAT32}, this scalar must be of the type {@link
     * ANEURALNETOWORKS_FLOAT32}, otherwise if all the input tensors have the
     * type {@link ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type
     * {@link ANEURALNETWORKS_FLOAT16}.
     * * 50: The clipping threshold for the output from the
     *       projection layer, such that values are bound within
     *       [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     *       If all the input tensors have type {@link
     * ANEURALNETWORKS_TENSOR_FLOAT32}, this scalar must be of the type {@link
     * ANEURALNETOWORKS_FLOAT32}, otherwise if all the input tensors have the
     * type {@link ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type
     * {@link ANEURALNETWORKS_FLOAT16}.
     * * 51: merge_outputs
     *       An {@link ANEURALNETWORKS_BOOL} scalar specifying if the outputs
     *       from forward and backward cells should be merged.
     * * 52: time_major
     *       An {@link ANEURALNETWORKS_BOOL} scalar specifying the shape format
     *       of input and output tensors.
     *
     * Outputs:
     * * 0: The forward output.
     *      A 3-D tensor of shape:
     *        If time-major: [max_time, batch_size, output_size]
     *        If batch-major: [batch_size, max_time, output_size]
     * * 1: The backward output.  Unused if merge_outputs is true.
     *      A 3-D tensor of shape:
     *        If time-major: [max_time, batch_size, output_size]
     *        If batch-major: [batch_size, max_time, output_size]
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM = 42,
    /**
     * A recurrent neural network layer that applies a basic RNN cell to a
     * sequence of inputs in forward and backward directions.
     *
     * This Op unrolls the input along the sequence dimension, and implements
     * the following operation for each element in the sequence s =
     * 1...sequence_length:
     *   fw_outputs[s] = fw_state = activation(inputs[s] * fw_input_weights’ +
     *          fw_state * fw_recurrent_weights’ + fw_bias)
     *
     * And for each element in sequence t = sequence_length : 1
     *   bw_outputs[t] = bw_state = activation(inputs[t] * bw_input_weights’ +
     *          bw_state * bw_recurrent_weights’ + bw_bias)
     *
     * Where:
     * * “{fw,bw}_input_weights” is a weight matrix that multiplies the inputs;
     * * “{fw,bw}_recurrent_weights” is a weight matrix that multiplies the
     *    current “state” which itself is the output from the previous time step
     *    computation;
     * * “{fw,bw}_bias” is a bias vector (added to each output vector in the
     *    batch);
     * * “activation” is the function passed as the “fused_activation_function”
     *   argument (if not “NONE”).
     *
     * The op also supports an auxiliary input. Regular cell feeds one input
     * into the two RNN cells in the following way:
     *
     *       INPUT  (INPUT_REVERSED)
     *         |         |
     *    ---------------------
     *    | FW_RNN     BW_RNN |
     *    ---------------------
     *         |         |
     *      FW_OUT     BW_OUT
     *
     * An op with an auxiliary input takes two inputs and feeds them into the
     * RNN cells in the following way:
     *
     *       AUX_INPUT   (AUX_INPUT_REVERSED)
     *           |             |
     *     INPUT | (INPUT_R'D.)|
     *       |   |       |     |
     *    -----------------------
     *    |  \  /        \    / |
     *    | FW_RNN       BW_RNN |
     *    -----------------------
     *         |           |
     *      FW_OUT      BW_OUT
     *
     * While stacking this op on top of itself, this allows to connect both
     * forward and backward outputs from previous cell to the next cell's
     * inputs.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * The input tensors must all be the same type.
     *
     * Inputs:
     * * 0: input.
     *      A 3-D tensor. The shape is defined by the input 6 (timeMajor). If
     *      it is set to true, then the input has a shape [maxTime, batchSize,
     *      inputSize], otherwise the input has a shape [batchSize, maxTime,
     *      inputSize].
     * * 1: fwWeights.
     *      A 2-D tensor of shape [fwNumUnits, inputSize].
     * * 2: fwRecurrentWeights.
     *      A 2-D tensor of shape [fwNumUnits, fwNumUnits].
     * * 3: fwBias.
     *      A 1-D tensor of shape [fwNumUnits].
     * * 4: fwHiddenState.
     *      A 2-D tensor of shape [batchSize, fwNumUnits]. Specifies a hidden
     *      state input for the first time step of the computation.
     * * 5: bwWeights.
     *      A 2-D tensor of shape [bwNumUnits, inputSize].
     * * 6: bwRecurrentWeights.
     *      A 2-D tensor of shape [bwNumUnits, bwNumUnits].
     * * 7: bwBias.
     *      A 1-D tensor of shape [bwNumUnits].
     * * 8: bwHiddenState
     *      A 2-D tensor of shape [batchSize, bwNumUnits]. Specifies a hidden
     *      state input for the first time step of the computation.
     * * 9: auxInput.
     *      A 3-D tensor. The shape is the same as of the input 0.
     * * 10:fwAuxWeights.
     *      A 2-D tensor of shape [fwNumUnits, inputSize].
     * * 11:bwAuxWeights.
     *      A 2-D tensor of shape [bwNumUnits, inputSize].
     * * 12:fusedActivationFunction.
     *      A {@link FuseCode} value indicating the activation function. If
     *      “NONE” is specified then it results in a linear activation.
     * * 13:timeMajor
     *      An {@link ANEURALNETWORKS_BOOL} scalar specifying the shape format
     *      of input and output tensors.
     * * 14:mergeOutputs
     *      An {@link ANEURALNETWORKS_BOOL} scalar specifying if the outputs
     *      from forward and backward cells are separate (if set to false) or
     *      concatenated (if set to true).
     * Outputs:
     * * 0: fwOutput.
     *      A 3-D tensor. The first two dimensions of the shape are defined by
     *      the input 6 (timeMajor) and the third dimension is defined by the
     *      input 14 (mergeOutputs). If timeMajor is set to true, then the first
     *      two dimensions are [maxTime, batchSize], otherwise they are set to
     *      [batchSize, maxTime]. If mergeOutputs is set to true, then the third
     *      dimension is equal to (fwNumUnits + bwNumUnits), otherwise it is set
     *      to fwNumUnits.
     * * 1: bwOutput.
     *      A 3-D tensor. If the input 14 (mergeOutputs) is set to true, then
     *      this tensor is not produced. The shape is defined by the input 6
     *      (timeMajor). If it is set to true, then the shape is set to
     *      [maxTime, batchSize, bwNumUnits], otherwise the shape is set to
     *      [batchSize, maxTime, bwNumUnits].
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_RNN = 43,
    /**
     * Greedily selects a subset of bounding boxes in descending order of score.
     *
     * This op applies hard NMS algorithm to each class. In each loop of
     * execution, the box with maximum score gets selected, and any boxes with
     * the intersection-over-union (IOU) greater than a threshold are removed
     * from the pending set.
     *
     * Axis-aligned bounding boxes are represented by its upper-left corner
     * coordinate (x1,y1) and lower-right corner coordinate (x2,y2). A valid
     * bounding box should satisfy x1 <= x2 and y1 <= y2.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Inputs:
     * * 0: A 2-D Tensor of shape [num_rois, num_classes], specifying the score
     *      of each bounding box proposal. The boxes are grouped by batches in
     * the first dimension.
     * * 1: A 2-D Tensor specifying the bounding boxes of shape
     *      [num_rois, num_classes * 4], organized in the order [x1, y1, x2,
     * y2]. The boxes are grouped by batches in the first dimension. The
     * sequential order of the boxes corresponds with input0. For input0 of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, this tensor should be
     * of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, with zeroPoint of 0
     * and scale of 0.125.
     * * 2: A 1-D Tensor of shape [batches], specifying the number of boxes
     *      for each image in the batch.
     * * 3: An {@link ANEURALNETWORKS_FLOAT32} scalar, score_threshold. Boxes
     *      with scores lower than the threshold are filtered before sending
     *      to the NMS algorithm.
     * * 4: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the IoU
     *      threshold.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the maximum
     *      number of selected bounding boxes for each image. Set to a negative
     *      value for unlimited number of output bounding boxes.
     *
     * Outputs:
     * * 0: A 1-D Tensor of the same {@link OperandCode} as input0, with shape
     *      [num_output_rois], specifying the score of each output box. The
     * boxes are grouped by batches, but the sequential order in each batch is
     * not guaranteed. For type of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the scale and zero point must be the same as input0.
     * * 1: A 2-D Tensor of the same {@link OperandCode} as input1, with shape
     *      [num_output_rois, 4], specifying the coordinates of each
     *      output bounding box with the same format as input1. The sequential
     *      order of the boxes corresponds with output0. For type of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, the scale must be
     *      0.125 and the zero point must be 0.
     * * 2: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     *      [num_output_rois], specifying the class of each output box. The
     *      sequential order of the boxes corresponds with output0.
     * * 3: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     *      [batches], specifying the number of output boxes for each image.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_BOX_WITH_NMS_LIMIT = 44,
    /**
     * Casts a tensor to a new type.
     *
     * This operation ignores the scale and zeroPoint of quanized tensors,
     * e.g. it treats a {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} input
     * as a tensor of uint8 values.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: A tensor with the same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_CAST = 45,
    /**
     * Shuffle the channels of the input tensor.
     *
     * Given an input tensor and a integer value of num_groups, CHANNEL_SHUFFLE
     * divide the channel dimension into num_groups groups, and reorganize the
     * channels by grouping channels with the same index in each group.
     *
     * Along the channel dimension, the output is calculated using this formula:
     *
     *     output_channel[k * num_groups + g] = input_channel[g * group_size +
     * k]
     *
     * where group_size = num_channels / num_groups
     *
     * The number of channels must be divisible by num_groups.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be shuffled.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
     *      groups.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the dimension
     *      channel shuffle would be performed on. Negative index is used to
     *      specify axis from the end (e.g. -1 for the last axis). Must be in
     *      the range [-n, n).
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} and same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_CHANNEL_SHUFFLE = 46,
    /**
     * Apply postprocessing steps to bounding box detections.
     *
     * Bounding box detections are generated by applying transformation on a set
     * of predefined anchors with the bounding box deltas from bounding box
     * regression. A final step of hard NMS is applied to limit the number of
     * returned boxes.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Inputs:
     * * 0: A 3-D Tensor of shape [batches, num_anchors, num_classes],
     * specifying the score of each anchor with each class. Class 0 for each
     *      [batches, num_anchors, 0] is background and will be ignored.
     * * 1: A 3-D Tensor of shape [batches, num_anchors, length_box_encoding],
     * with the first four values in length_box_encoding specifying the bounding
     *      box deltas. The box deltas are encoded in the order of [dy, dx, dh,
     * dw], where dy and dx is the linear-scale relative correction factor for
     * the center position of the bounding box with respect to the width and
     * height, dh and dw is the log-scale relative correction factor for the
     * width and height. All the entries in length_box_encoding beyond the first
     * four values are ignored in this operation.
     * * 2: A 2-D Tensor of shape [num_anchors, 4], specifying the shape of each
     *      predefined anchor, with format [ctr_y, ctr_x, h, w], where ctr_y and
     *      ctr_x are the center position of the box, and h and w are the height
     *      and the width.
     * * 3: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the scaling
     *      factor for dy in bounding box deltas.
     * * 4: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the scaling
     *      factor for dx in bounding box deltas.
     * * 5: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the scaling
     *      factor for dh in bounding box deltas.
     * * 6: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the scaling
     *      factor for dw in bounding box deltas.
     * * 7: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to use regular
     *      multi-class NMS algorithm that do NMS separately for each class,
     *      set to false for a faster algorithm that only do one single NMS
     *      using the highest class score..
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, max_num_detections,
     * specifying the maximum number of boxes for the output. Boxes with the
     * lowest scores are discarded to meet the limit.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, only used when input7 is
     *      set to false, specifying the maximum number of classes per
     * detection.
     * * 10: An {@link ANEURALNETWORKS_INT32} scalar, only used when input7 is
     *       set to true, specifying the maximum number of detections when
     *       applying NMS algorithm for each single class.
     * * 11: An {@link ANEURALNETWORKS_FLOAT32} scalar, score_threshold. Boxes
     *       with scores lower than the threshold are filtered before sending
     *       to the NMS algorithm.
     * * 12: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the IoU
     *       threshold for hard NMS.
     * * 13: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to include
     *       background class in the list of label map for the output, set
     *       to false to not include the background. When the background
     *       class is included, it has label 0 and the output classes start
     *       at 1 in the label map, otherwise, the output classes start at 0.
     *
     * Outputs:
     * * 0: A 2-D tensor of the same {@link OperandCode} as input0, with shape
     *      [batches, max_num_detections], specifying the score of each output
     *      detections.
     * * 1: A 3-D tensor of shape [batches, max_num_detections, 4], specifying
     * the coordinates of each output bounding box, with format [y1, x1, y2,
     * x2].
     * * 2: A 2-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     *      [batches, max_num_detections], specifying the class label for each
     *      output detection.
     * * 3: An 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     * [batches], specifying the number of valid output detections for each
     * batch.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_DETECTION_POSTPROCESSING = 47,
    /**
     * For input tensors x and y, computes x == y elementwise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and dimensions compatible
     *      with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_EQUAL = 48,
    /**
     * Computes exponential of x element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_EXP = 49,
    /**
     * Inserts a dimension of 1 into a tensor's shape.
     *
     * Given a tensor input, this operation inserts a dimension of 1 at the
     * given dimension index of input's shape. The dimension index starts at
     * zero; if you specify a negative dimension index, it is counted backward
     * from the end.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: An n-D tensor.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar specifying the dimension
     *      index to expand. Must be in the range [-(n + 1), (n + 1)).
     *
     * Outputs:
     * * 0: An (n + 1)-D tensor with the same {@link OperandCode} and data as
     *      input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_EXPAND_DIMS = 50,
    /**
     * Gathers values along an axis.
     *
     * Produces an output tensor with shape
     *     input0.dimension[:axis] + indices.dimension + input0.dimension[axis +
     * 1:] where: # Vector indices (output is rank(input0)). output[a_0, ...,
     * a_n, i, b_0, ..., b_n] = input0[a_0, ..., a_n, indices[i], b_0, ..., b_n]
     *
     *     # Higher rank indices (output is rank(input0) + rank(indices) - 1).
     *     output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
     *       input0[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: An n-D tensor from which to gather values.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar specifying the axis.
     *      Negative index is used to specify axis from the end
     *      (e.g. -1 for the last axis). Must be in the range [-n, n).
     * * 2: A k-D tensor {@link ANEURALNETWORKS_TENSOR_INT32} of indices.
     *      The values must be in the bounds of the corresponding dimensions
     *      of input0.
     *
     * Outputs:
     * * 0: An (n + k - 1)-D tensor with the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_GATHER = 51,
    /**
     * Generate aixs-aligned bounding box proposals.
     *
     * Bounding box proposals are generated by applying transformation on a set
     * of predefined anchors with the bounding box deltas from bounding box
     * regression. A final step of hard NMS is applied to limit the number of
     * returned boxes.
     *
     * Axis-aligned bounding boxes are represented by its upper-left corner
     * coordinate (x1,y1) and lower-right corner coordinate (x2,y2). A valid
     * bounding box should satisfy x1 <= x2 and y1 <= y2.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Inputs:
     * * 0: A 4-D Tensor specifying the score of each anchor at each
     *      location. With "NHWC" data layout, the tensor shape is
     *      [batches, height, width, num_anchors]. With "NCHW" data layout,
     *      the tensor shape is [batches, num_anchors, height, width].
     * * 1: A 4-D Tensor specifying the bounding box deltas. With "NHWC" data
     *      layout, the tensor shape is [batches, height, width, num_anchors *
     * 4]. With "NCHW" data layout, the tensor shape is [batches, num_anchors *
     * 4, height, width]. The box deltas are encoded in the order of [dx, dy,
     * dw, dh], where dx and dy is the linear-scale relative correction factor
     * for the center position of the bounding box with respect to the width and
     * height, dw and dh is the log-scale relative correction factor for the
     * width and height. The last dimensions is the channel dimension.
     * * 2: A 2-D Tensor of shape [num_anchors, 4], specifying the shape of each
     *      predefined anchor, with format [x1, y1, x2, y2]. For input0 of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, this tensor should be
     * of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT16_SYMM}, with scale of 0.125.
     * * 3: A 2-D Tensor of shape [batches, 2], specifying the size of
     *      each image in the batch, with format [image_height, image_width].
     * * 4: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
     *      from the height of original image to the height of feature map.
     * * 5: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
     *      from the width of original image to the width of feature map.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the maximum
     *      number of boxes before going into the hard NMS algorithm. Boxes
     *      with the lowest scores are discarded to meet the limit. Set to
     *      a non-positive value for unlimited number.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the maximum
     *      number of boxes returning from the hard NMS algorithm. Boxes
     *      with the lowest scores are discarded to meet the limit. Set to
     *      a non-positive value for unlimited number.
     * * 8: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the IoU
     *      threshold for hard NMS.
     * * 9: An {@link ANEURALNETWORKS_FLOAT32} scalar, min_size. Boxes with
     *      height or width lower than the absolute threshold are filtered out.
     * * 10: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *       NCHW data layout for input0 and input1. Set to false for NHWC.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0, of shape
     *      [num_output_rois], specifying the score of each output box.
     *      The boxes are grouped by batches, but the sequential order in
     *      each batch is not guaranteed. For type of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the scale and zero
     *      point must be the same as input0.
     * * 1: A tensor of the same {@link OperandCode} as input1, of shape
     *      [num_output_rois, 4], specifying the coordinates of each output
     *      bounding box for each class, with format [x1, y1, x2, y2].
     *      The sequential order of the boxes corresponds with output0.
     *      For type of {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, the
     *      scale must be 0.125 and the zero point must be 0.
     * * 2: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     *      [batches], specifying the number of output boxes for each image.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_GENERATE_PROPOSALS = 52,
    /**
     * For input tensors x and y, computes x > y elementwise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and dimensions compatible
     *      with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_GREATER = 53,
    /**
     * For input tensors x and y, computes x >= y elementwise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and dimensions compatible
     *      with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_GREATER_EQUAL = 54,
    /**
     * Performs a grouped 2-D convolution operation.
     *
     * Given an input tensor of shape [batches, height, width, depth_in] and a
     * filter tensor of shape [depth_out, filter_height, filter_width,
     depth_group]
     * containing depth_out convolutional filters of depth depth_group,
     GROUPED_CONV
     * applies a group of different filters to each input channel group, then
     * concatenates the results together.
     *
     * Specifically, the input channels are divided into num_groups groups, each
     with
     * depth depth_group, i.e. depth_in = num_groups * depth_group. The
     convolutional
     * filters are also divided into num_groups groups, i.e. depth_out is
     divisible
     * by num_groups. GROUPED_CONV applies each group of filters to the
     corresponding
     * input channel group, and the result are concatenated together.
     *
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, i, j, g * channel_multiplier + q] =
     *         sum_{di, dj, dk} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj,
     *                   g * depth_group + dk] *
     *             filter[g * channel_multiplier + q, di, dj, dk]
     *         ) + bias[channel]
     *
     * where channel_multiplier = depth_out / num_groups
     *
     * Supported tensor {@link OperandCode} configurations:
     * * 32 bit Floating point :
     * * * {@link ANEURALNETWORKS_TENSOR_FLOAT32} for input, filter, output, and
     bias.
     *
     * * 16 bit Floating point:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} for input, filter, output, and
     bias.
     *
     * * Quantized:
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} for input, filter, and
     output.
     * * * {@link ANEURALNETWORKS_TENSOR_INT32} for bias (with scale set to
     * * * input.scale * filter.scale).
     *
     * * Quantized with symetric per channel quantization for the filter:
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} for input, and output.
     * * * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} for filter.
     * * * {@link ANEURALNETWORKS_TENSOR_INT32} for bias (scale set to 0.0,
     * * * each value scaling is separate and equal to input.scale *
     filter.scales[channel]).
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both explicit padding and implicit padding are supported.
     *
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input, where depth_in = num_groups * depth_group.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_group], specifying
     *      the filter, where depth_out must be divisible by num_groups.  For
     *      tensor of type {@link
     ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}
     *      the channel dimension (channelDim at
     *      {@link ANeuralNetworksSymmPerChannelQuantParams}) must be set to 0.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias must be of the same
     *      type. For filter tensor of {@link
     ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the bias should be of {@link ANEURALNETWORKS_TENSOR_INT32}, with
     zeroPoint
     *      of 0 and bias_scale == input_scale * filter_scale. For filter tensor
     *      of {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}, the bias
     *      should be of {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of
     *      0 and bias_scale of 0. The actual scale of each value 'i' is equal
     to
     *      bias_scale[i] = input_scale * filter_scale[i].
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
            groups.
     * * 10: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *       {@link FuseCode} values. Specifies the activation to
     *       invoke on the result.
     * * 11: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *       NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input, where depth_in = num_groups * depth_group.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_group], specifying
     *      the filter, where depth_out must be divisible by num_groups.  For
     *      tensor of type {@link
     ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}
     *      the channel dimension (channelDim at
     *      {@link ANeuralNetworksSymmPerChannelQuantParams}) must be set to 0.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias must be of the same
     *      type. For filter tensor of {@link
     ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the bias should be of {@link ANEURALNETWORKS_TENSOR_INT32}, with
     zeroPoint
     *      of 0 and bias_scale == input_scale * filter_scale. For filter tensor
     *      of {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}, the bias
     *      should be of {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of
     *      0 and bias_scale of 0. The actual scale of each value 'i' is equal
     to
     *      bias_scale[i] = input_scale * filter_scale[i].
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      {@link PaddingCode} values.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
     *      groups.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 8: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *      NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth_out]. For output tensor of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the following condition
     *      must be satisfied: output_scale > input_scale * filter_scale (for
     *      filter tensor of type {@link
     ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}
     *      this condition must be true for all filter scales).
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_GROUPED_CONV_2D = 55,
    /**
     * Localize the maximum keypoints from heatmaps.
     *
     * This operation approximates the accurate maximum keypoint scores and
     * indices after bicubic upscaling by using Taylor expansion up to the
     * quadratic term.
     *
     * The bounding box is represented by its upper-left corner coordinate
     * (x1,y1) and lower-right corner coordinate (x2,y2) in the original image.
     * A valid bounding box should satisfy x1 <= x2 and y1 <= y2.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: A 4-D Tensor of shape
     *      [num_boxes, heatmap_size, heatmap_size, num_keypoints],
     *      specifying the heatmaps, the height and width of heatmaps should
     *      be the same, and must be greater than or equal to 2.
     * * 1: A 2-D Tensor of shape [num_boxes, 4], specifying the bounding boxes,
     *      each with format [x1, y1, x2, y2]. For input0 of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, this tensor should
     *      be of {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, with zeroPoint
     *      of 0 and scale of 0.125.
     * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *      NCHW data layout for input0. Set to false for NHWC.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0, with shape
     *      [num_boxes, num_keypoints], specifying score of the keypoints.
     * * 1: A tensor of the same {@link OperandCode} as input1, with shape
     *      [num_boxes, num_keypoints, 2], specifying the location of
     *      the keypoints, the second dimension is organized as
     *      [keypoint_x, keypoint_y].
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_HEATMAP_MAX_KEYPOINT = 56,
    /**
     * Applies instance normalization to the input tensor.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, h, w, c] =
     *         (input[b, h, w, c] - mean[b, c]) * gamma /
     *         sqrt(var[b, c] + epsilon) + beta
     *
     * Where the mean and variance are computed across the spatial dimensions:
     *
     *     mean[b, c] =
     *         sum_{h, w}(input[b, h, w, c]) / sum(1)
     *
     *     var[b, c] =
     *         sum_{h, w}(pow(input[b, h, w, c] - mean[b, c], 2)) / sum(1)
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be normalized.
     * * 1: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying gamma, the
     *      scale applied to the normalized tensor.
     * * 2: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying beta, the
     *      offset applied to the normalized tensor.
     * * 3: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying epsilon, the
     *      small value added to variance to avoid dividing by zero.
     * * 4: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *      NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} and same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_INSTANCE_NORMALIZATION = 57,
    /**
     * For input tensors x and y, computes x < y elementwise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and dimensions compatible
     *      with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_LESS = 58,
    /**
     * For input tensors x and y, computes x <= y elementwise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and dimensions compatible
     *      with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_LESS_EQUAL = 59,
    /**
     * Computes natural logarithm of x element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_LOG = 60,
    /**
     * Returns the truth value of x AND y element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     * * 1: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8} and dimensions
     *      compatible with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_LOGICAL_AND = 61,
    /**
     * Computes the truth value of NOT x element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_LOGICAL_NOT = 62,
    /**
     * Returns the truth value of x OR y element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     * * 1: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8} and dimensions
     *      compatible with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_LOGICAL_OR = 63,
    /**
     * Computes the log softmax activations given logits.
     *
     * The output is calculated using this formula:
     *
     *     output = logits * beta - log(reduce_sum(exp(logits * beta), axis))
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor specifying the input logits.
     * * 1: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the positive
     *      scaling factor for the exponent, beta.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar specifying the axis to
     *      reduce across. Negative index is used to specify axis from the
     *      end (e.g. -1 for the last axis). Must be in the range [-n, n).
     *
     * Outputs:
     * * 0: The output tensor of the same {@link OperandCode} and shape as
     *      input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_LOG_SOFTMAX = 64,
    /**
     * Returns the element-wise maximum of two tensors.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and compatible dimensions
     *      with input0.
     *
     * Outputs:
     * * 0: The sum, a tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_MAXIMUM = 65,
    /**
     * Returns the element-wise minimum of two tensors.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and compatible dimensions
     *      with input0.
     *
     * Outputs:
     * * 0: The sum, a tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_MINIMUM = 66,
    /**
     * Computes numerical negative value element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_NEG = 67,
    /**
     * For input tensors x and y, computes x != y elementwise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * This operation supports broadcasting.
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandCode} and dimensions compatible
     *      with input0.
     *
     * Outputs:
     * * 0: A tensor of {@link ANEURALNETWORKS_TENSOR_BOOL8}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_NOT_EQUAL = 68,
    /**
     * Pads a tensor with the given constant value according to the specified
     * paddings.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be padded.
     * * 1: A 2-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the paddings
     *      for each spatial dimension of the input tensor. The shape of the
     *      tensor must be {rank(input0), 2}.
     *      padding[i, 0] specifies the number of elements to be padded in the
     *      front of dimension i.
     *      padding[i, 1] specifies the number of elements to be padded after
     *      the end of dimension i.
     * * 2: An scalar specifying the value to use for padding input0.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32}, the
     *      pad value should be of {@link ANEURALNETWORKS_FLOAT32}.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      the pad value should be of {@link ANEURALNETWORKS_INT32}. The
     *      scale and zeroPoint are assumed to be the same as in input0.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0. The
     *      output tensor has the same rank as input0, and each
     *      dimension of the output tensor has the same size as the
     *      corresponding dimension of the input tensor plus the size
     *      of the padding:
     *          output0.dimension[i] =
     *              padding[i, 0] + input0.dimension[i] + padding[i, 1]
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_PAD_V2 = 69,
    /**
     * Computes the power of one value to another.
     *
     * Given a tensor base and a tensor exponent, this operation computes
     * base^exponent elementwise.
     *
     * This operations supports broadcasting. The size of the output is the
     * maximum size along each dimension of the input operands. It starts with
     * the trailing dimensions, and works its way forward.
     *
     * For example:
     *     base.dimension     =    {4, 1, 2}
     *     exponent.dimension = {5, 4, 3, 1}
     *     output.dimension   = {5, 4, 3, 2}
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: A tensor specifying the base.
     * * 1: A tensor specifying the exponent.
     *
     * Outputs:
     * * 0: An output tensor.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_POW = 70,
    /**
     * Parametric Rectified Linear Unit.
     *
     * It follows: f(x) = alpha * x for x < 0, f(x) = x for x >= 0, where alpha
     * is a learned array with the same {@link OperandCode} and compatible
     * dimensions as input x.
     *
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     *
     * The size of the output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its way
     * forward.
     *
     * Example:
     *     input.dimension  =    {4, 1, 2}
     *     alpha.dimension  = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     * * 1: A tensor of the same {@link OperandCode}, and compatible dimensions
     *      as input0, specifying the alpha.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_PRELU = 71,
    /**
     * Quantizes the input tensor.
     *
     * The formula is:
     *
     *     output = max(0, min(255, round(input / scale) + zeroPoint)
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0, but with
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_QUANTIZE = 72,
    /**
     * A version of quantized LSTM, using 16 bit quantization for internal
     * state.
     *
     * There is no projection layer, so cell state size is equal to the output
     * size.
     *
     * Inputs:
     * * 0: A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [numBatches, inputSize] specifying the input to the LSTM
     *      cell. Tensor is quantized with a fixed quantization range of
     *      [-1, 127/128] (scale = 1/128, zeroPoint = 128).
     * * 1: The input-to-input weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying input-to-input part of
     *      weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 2: The input-to-forget weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying input-to-forget part of
     *      weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 3: The input-to-cell weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying input-to-cell part of
     *      weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 4: The input-to-output weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying input-to-output part of
     *      weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 5: The recurrent-to-input weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying recurrent-to-input part
     *      of weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 6: The recurrent-to-forget weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying recurrent-to-forget
     *      part of weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 7: The recurrent-to-cell weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying recurrent-to-cell part
     *      of weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 8: The recurrent-to-output weights.
     *      A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [outputSize, inputSize] specifying recurrent-to-output
     *      part of weights for fully-connected layer inside the LSTM cell.
     *      Quantization zero point and scale must be the same across all the
     *      weights.
     * * 9: The input gate bias.
     *      A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32} and shape
     *      [outputSize] specifying the bias for the fully-connected layer
     *      inside the LSTM cell. Bias is quantized with scale being a product
     *      of input and weights scales and zeroPoint equal to 0.
     * * 10:The forget gate bias.
     *      A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32} and shape
     *      [outputSize] specifying the bias for the fully-connected layer
     *      inside the LSTM cell. Bias is quantized with scale being a product
     *      of input and weights scales and zeroPoint equal to 0.
     * * 11:The cell bias.
     *      A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32} and shape
     *      [outputSize] specifying the bias for the fully-connected layer
     *      inside the LSTM cell. Bias is quantized with scale being a product
     *      of input and weights scales and zeroPoint equal to 0.
     * * 12:The output gate bias.
     *      A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32} and shape
     *      [outputSize] specifying the bias for the fully-connected layer
     *      inside the LSTM cell. Bias is quantized with scale being a product
     *      of input and weights scales and zeroPoint equal to 0.
     * * 13: A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT16_SYMM}
     *       and shape [numBatches, outputSize] specifying the cell state from
     * the previous time step of the LSTM cell. It is quantized using a
     *       quantization range of [-2^4, 2^4 * 32767/32768] (scale = 2^4 /
     *       32768, zeroPoint = 0).
     * * 14: A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *       and shape [numBathes, outputSize] specifying the output of the LSTM
     *       cell from previous time-step. Tensor is quantized with a fixed
     *       quantization range of [-1, 127/128] (scale = 1/128, zeroPoint =
     *       128).
     *
     *
     * Outputs:
     * * 0: A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT16_SYMM}
     *      and shape [numBatches, outputSize] which contains a cell state from
     *      the current time step. Tensor is quantized using a quantization
     *      range of [-2^4, 2^4 * 32767/32768] (scale = 2^4 / 32768, zeroPoint =
     *      0).
     * * 1: A 2-D tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *      and shape [numBathes, outputSize] which contains the output value.
     *      Tensor is quantized with a fixed quantization range of [-1, 127/128]
     *      (scale = 1/128, zeroPoint = 128).
     */
    ANEURALNETWORKS_QUANTIZED_16BIT_LSTM = 73,
    /**
     * Draws samples from a multinomial distribution.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Inputs:
     * * 0: A 2-D tensor with shape [batches, classes], specifying the
     *      unnormalized log-probabilities for all classes.
     * * 1: A scalar {@link ANEURALNETWORKS_INT32}, specifying the number of
     *      independent samples to draw for each row slice.
     * * 2: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor with shape [2],
     *      specifying seeds used to initialize the random distribution.
     * Outputs:
     * * 0: A 2-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor with shape
     *      [batches, samples], containing the drawn samples.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_RANDOM_MULTINOMIAL = 74,
    /**
     * Reduces a tensor by computing the "logical and" of elements along given
     * dimensions.
     *
     * If keep_dims is true, the reduced dimensions are
     * retained with length 1. Otherwise, the rank of the tensor is reduced by
     * 1 for each entry in dimensions.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor.
     * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions
     *      to reduce. Dimension values must be in the range [-n, n).
     * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, keep_dims. If true,
     *      retains reduced dimensions with length 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_REDUCE_ALL = 75,
    /**
     * Reduces a tensor by computing the "logical or" of elements along given
     * dimensions.
     *
     * If keep_dims is true, the reduced dimensions are
     * retained with length 1. Otherwise, the rank of the tensor is reduced by
     * 1 for each entry in dimensions.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_BOOL8}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor.
     * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions
     *      to reduce. Dimension values must be in the range [-n, n).
     * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, keep_dims. If true,
     *      retains reduced dimensions with length 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_REDUCE_ANY = 76,
    /**
     * Reduces a tensor by computing the maximum of elements along given
     * dimensions.
     *
     * If keep_dims is true, the reduced dimensions are
     * retained with length 1. Otherwise, the rank of the tensor is reduced by
     * 1 for each entry in dimensions.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor.
     * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions
     *      to reduce. Dimension values must be in the range [-n, n).
     * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, keep_dims. If true,
     *      retains reduced dimensions with length 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_REDUCE_MAX = 77,
    /**
     * Reduces a tensor by computing the minimum of elements along given
     * dimensions.
     *
     * If keep_dims is true, the reduced dimensions are
     * retained with length 1. Otherwise, the rank of the tensor is reduced by
     * 1 for each entry in dimensions.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor.
     * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions
     *      to reduce. Dimension values must be in the range [-n, n).
     * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, keep_dims. If true,
     *      retains reduced dimensions with length 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_REDUCE_MIN = 78,
    /**
     * Reduces a tensor by multiplying elements along given dimensions.
     *
     * If keep_dims is true, the reduced dimensions are
     * retained with length 1. Otherwise, the rank of the tensor is reduced by
     * 1 for each entry in dimensions.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor.
     * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions
     *      to reduce. Dimension values must be in the range [-n, n).
     * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, keep_dims. If true,
     *      retains reduced dimensions with length 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_REDUCE_PROD = 79,
    /**
     * Reduces a tensor by summing elements along given dimensions.
     *
     * If keep_dims is true, the reduced dimensions are
     * retained with length 1. Otherwise, the rank of the tensor is reduced by
     * 1 for each entry in dimensions.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: An n-D tensor.
     * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions
     *      to reduce. Dimension values must be in the range [-n, n).
     * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, keep_dims. If true,
     *      retains reduced dimensions with length 1.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_REDUCE_SUM = 80,
    /**
     * Select and scale the feature map of each region of interest to a unified
     * output size by average pooling sampling points from bilinear
     * interpolation.
     *
     * The region of interest is represented by its upper-left corner coordinate
     * (x1,y1) and lower-right corner coordinate (x2,y2) in the original image.
     * A spatial scaling factor is applied to map into feature map coordinate.
     * A valid region of interest should satisfy x1 <= x2 and y1 <= y2.
     *
     * No rounding is applied in this operation. The sampling points are unified
     * distributed in the pooling bin and their values are calculated by
     * bilinear interpolation.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: A 4-D tensor, specifying the feature map.
     * * 1: A 2-D Tensor of shape [num_rois, 4], specifying the locations of
     *      the regions of interest, each line with format [x1, y1, x2, y2].
     *      For input0 of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      this tensor should be of {@link
     * ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, with zeroPoint of 0 and scale of
     * 0.125.
     * * 2: An 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     *      [batches], specifying the number of output boxes for each batch.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      height of the output tensor.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      width of the output tensor.
     * * 5: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
     *      from the height of original image to the height of feature map.
     * * 6: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
     *      from the width of original image to the width of feature map.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
     *      sampling points in height dimension used to compute the output.
     *      Set to 0 for adaptive value of ceil(roi_height/out_height).
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
     *      sampling points in width dimension used to compute the output.
     *      Set to 0 for adaptive value of ceil(roi_width/out_width).
     * * 9: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *      NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0. The output
     *      shape is [num_rois, out_height, out_width, depth].
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_ROI_ALIGN = 81,
    /**
     * Select and scale the feature map of each region of interest to a unified
     * output size by max-pooling.
     *
     * The region of interest is represented by its upper-left corner coordinate
     * (x1,y1) and lower-right corner coordinate (x2,y2) in the original image.
     * A spatial scaling factor is applied to map into feature map coordinate.
     * A valid region of interest should satisfy x1 <= x2 and y1 <= y2.
     *
     * Rounding is applied in this operation to ensure integer boundary for
     * regions of interest and pooling bins.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: A 4-D tensor, specifying the feature map.
     * * 1: A 2-D Tensor of shape [num_rois, 4], specifying the locations of
     *      the regions of interest, each line with format [x1, y1, x2, y2].
     *      For input0 of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
     *      this tensor should be of {@link
     * ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, with zeroPoint of 0 and scale of
     * 0.125.
     * * 2: An 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
     *      [batches], specifying the number of output boxes for each batch.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      height of the output tensor.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      width of the output tensor.
     * * 5: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
     *      from the height of original image to the height of feature map.
     * * 6: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
     *      from the width of original image to the width of feature map.
     * * 7: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *      NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} as input0. The output
     *      shape is [num_rois, out_height, out_width, depth].
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_ROI_POOLING = 82,
    /**
     * Computes reciprocal of square root of x element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_RSQRT = 83,
    /**
     * Using a tensor of booleans c and input tensors x and y select values
     * elementwise from both input tensors:
     *
     * O[i] = C[i] ? x[i] : y[i].
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: A tensor of type {@link ANEURALNETWORKS_TENSOR_BOOL8} acting as a
     *      mask that chooses, based on the value at each element, whether the
     *      corresponding element in the output should be taken from input1 (if
     *      true) or input2 (if false).
     * * 1: An input tensor of the same shape as input0.
     * * 2: An input tensor of the same shape and type as input1.
     *
     * Outputs:
     * * 0: A tensor of the same type and shape as input1 and input2.
     *
     */
    ANEURALNETWORKS_SELECT = 84,
    /**
     * Computes sin of x element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_SIN = 85,
    /**
     * Extracts a slice of specified size from the input tensor starting at a
     * specified location.
     *
     * The starting location is specified as a 1-D tensor containing offsets
     * for each dimension. The size is specified as a 1-D tensor containing
     * either size of a slice along corresponding dimension or -1. In the latter
     * case, all the remaining elements in dimension are included in the slice.
     * Slice size in each dimension cannot be zero.
     *
     * A sum of begin offset and a size of a slice must not exceed size of a
     * corresponding dimension.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: An n-D tensor to take slice from.
     * * 1: A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32} specifying
     *      the beginning indices of the slice in each dimension.
     * * 2: A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32} specifying
     *      the size of the slice in each dimension.
     *
     * Outputs:
     * * 0: An n-D tensor of the same type as the input containing the slice.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_SLICE = 86,
    /**
     * Splits a tensor along a given axis into num_splits subtensors.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: An n-D tensor to split.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar specifying the axis along
     *      which to split.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar indicating the number of
     *      splits along given axis. Must evenly divide axis size.
     *
     * Outputs:
     * * 0 ~ (num_splits - 1): Resulting subtensors.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_SPLIT = 87,
    /**
     * Computes square root of x element-wise.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: from 1.
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_SQRT = 88,
    /**
     * Constructs a tensor by tiling a given tensor.
     *
     * This operation creates a new tensor by replicating `input` `multiples`
     * times. The output tensor's i-th dimension has `input.dims(i) *
     * multiples[i]` elements, and the values of `input` are replicated
     * `multiples[i]` times along the i-th dimension. For example, tiling `[a b
     * c d]` by `[2]` produces `[a b c d a b c d]`.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: input, an n-D tensor specifying the input.
     * * 1: multiples, a 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}.
     *      The length of multiples must be n.
     *
     * Outputs:
     * * 0: A tiled tensor of the same {@link OperandCode} and rank as `input`.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TILE = 89,
    /**
     * Finds values and indices of the k largest entries for the last dimension.
     *
     * Resulting values in each dimensions are sorted in descending order. If
     * two values are equal, the one with larger index appears first.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_INT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: from 1
     *
     * Inputs:
     * * 0: input, an n-D tensor specifying the input.
     * * 1: k, an {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
     *      top elements to look for along the last dimension.
     *
     * Outputs:
     * * 0: An n-D tensor of the same type as the input, containing the k
     *      largest elements along each last dimensional slice.
     * * 1: An n-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32}
     *      containing the indices of values within the last dimension of input.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TOPK_V2 = 90,
    /**
     * Performs the tranpose of 2-D convolution operation.
     *
     * This operation is sometimes called "deconvolution" after Deconvolutional
     * Networks, but is actually the transpose (gradient) of
     * {@link ANEURALNETWORKS_CONV_2D} rather than an actual deconvolution.
     *
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both explicit padding and implicit padding are supported.
     *
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_in], specifying the
     *      filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias should be of the
     *      same type. For input tensor of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 9: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 10: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *       NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_in], specifying the
     *      filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of type {@link ANEURALNETWORKS_TENSOR_FLOAT32} or
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT16}, the bias should be of the
     *      same type. For input tensor of type
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link ANEURALNETWORKS_TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link ANEURALNETWORKS_TENSOR_INT32} tensor, specifying the
     * output tensor shape.
     * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      {@link PaddingCode} values.
     * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 6: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 7: An {@link ANEURALNETWORKS_INT32} scalar, and has to be one of the
     *      {@link FuseCode} values. Specifies the activation to
     *      invoke on the result.
     * * 8: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *      NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth_out]. For output tensor of
     *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, the following condition
     *      must be satisfied: output_scale > input_scale * filter_scale.
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_TRANSPOSE_CONV_2D = 91,
    /**
     * A recurrent neural network specified by an LSTM cell.
     *
     * Performs (fully) dynamic unrolling of input.
     *
     * This Op unrolls the input along the time dimension, and implements the
     * following operation for each element in the sequence
     * s = 1...sequence_length:
     *   outputs[s] = projection(state = activation(LSTMOp(inputs[s])))
     *
     * Where LSTMOp is the LSTM op as in {@link ANEURALNETWORKS_LSTM},
     * the "projection" is an optional projection layer from state and output
     * and the “activation” is the function passed as the
     * “fused_activation_function” argument (if not “NONE”).
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 3, either time-major or batch-major.
     *
     * All input and output tensors must be of the same type.
     *
     * Inputs:
     * * 0: The input (\f$x_t\f$).
     *      A 3-D tensor of shape:
     *        If time-major: [max_time, batch_size, output_size]
     *        If batch-major: [batch_size, max_time, output_size]
     *      where “max_size” is the number of timesteps (sequence length),
     *      “batch_size” corresponds to the batching dimension, and
     *      “input_size” is the size of the input.
     * * 1: The input-to-input weights (\f$W_{xi}\f$). Optional.
     *      A 2-D tensor of shape [num_units, input_size], where “num_units”
     *      corresponds to the number of cell units.
     * * 2: The input-to-forget weights (\f$W_{xf}\f$).
     *      A 2-D tensor of shape [num_units, input_size].
     * * 3: The input-to-cell weights (\f$W_{xc}\f$).
     *      A 2-D tensor of shape [num_units, input_size].
     * * 4: The input-to-output weights (\f$W_{xo}\f$).
     *      A 2-D tensor of shape [num_units, input_size].
     * * 5: The recurrent-to-input weights (\f$W_{hi}\f$). Optional.
     *      A 2-D tensor of shape [num_units, output_size], where “output_size”
     *      corresponds to either the number of cell units (i.e., “num_units”),
     *      or the second dimension of the “projection_weights”, if defined.
     * * 6: The recurrent-to-forget weights (\f$W_{hf}\f$).
     *      A 2-D tensor of shape [num_units, output_size].
     * * 7: The recurrent-to-cell weights (\f$W_{hc}\f$).
     *      A 2-D tensor of shape [num_units, output_size].
     * * 8: The recurrent-to-output weights (\f$W_{ho}\f$).
     *      A 2-D tensor of shape [num_units, output_size].
     * * 9: The cell-to-input weights (\f$W_{ci}\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 10:The cell-to-forget weights (\f$W_{cf}\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 11:The cell-to-output weights (\f$W_{co}\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 12:The input gate bias (\f$b_i\f$). Optional.
     *      A 1-D tensor of shape [num_units].
     * * 13:The forget gate bias (\f$b_f\f$).
     *      A 1-D tensor of shape [num_units].
     * * 14:The cell bias (\f$b_c\f$).
     *      A 1-D tensor of shape [num_units].
     * * 15:The output gate bias (\f$b_o\f$).
     *      A 1-D tensor of shape [num_units].
     * * 16:The projection weights (\f$W_{proj}\f$). Optional.
     *      A 2-D tensor of shape [output_size, num_units].
     * * 17:The projection bias (\f$b_{proj}\f$). Optional.
     *      A 1-D tensor of shape [output_size].
     * * 18:The output state (in) (\f$h_{t-1}\f$).
     *      A 2-D tensor of shape [batch_size, output_size].
     * * 19:The cell state (in) (\f$C_{t-1}\f$).
     *      A 2-D tensor of shape [batch_size, num_units].
     * * 20:The activation function (\f$g\f$).
     *      A value indicating the activation function:
     *      <ul>
     *      <li>0: None;
     *      <li>1: Relu;
     *      <li>3: Relu6;
     *      <li>4: Tanh;
     *      <li>6: Sigmoid.
     *      </ul>
     * * 21:The clipping threshold (\f$t_{cell}\f$) for the cell state, such
     *      that values are bound within [-cell_clip, cell_clip]. If set to 0.0
     *      then clipping is disabled.
     * * 22:The clipping threshold (\f$t_{proj}\f$) for the output from the
     *      projection layer, such that values are bound within
     *      [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     * * 23:Time-major if true, batch-major if false.
     * * 24:The input layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at input gate.
     * * 25:The forget layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at forget gate.
     * * 26:The cell layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at cell gate.
     * * 27:The output layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at output gate.
     *
     * Outputs:
     * * 0: The output (\f$o_t\f$).
     *      A 3-D tensor of shape:
     *        If time-major: [max_time, batch_size, output_size]
     *        If batch-major: [batch_size, max_time, output_size]
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM = 92,
    /**
     * A recurrent neural network layer that applies a basic RNN cell to a
     * sequence of inputs.
     *
     * This layer unrolls the input along the sequence dimension, and implements
     * the following operation
     * for each element in the sequence s = 1...sequence_length:
     *   outputs[s] = state = activation(inputs[s] * input_weights’ + state *
     *   recurrent_weights’ + bias)
     *
     * Where:
     * * “input_weights” is a weight matrix that multiplies the inputs;
     * * “recurrent_weights” is a weight matrix that multiplies the current
     *    “state” which itself is the output from the previous time step
     *    computation;
     * * “bias” is a bias vector (added to each output vector in the batch);
     * * “activation” is the function passed as the “fused_activation_function”
     *   argument (if not “NONE”).
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * The input tensors must all be the same type.
     *
     * Inputs:
     * * 0: input.
     *      A 3-D tensor. The shape is defined by the input 6 (timeMajor). If
     *      it is set to 1, then the input has a shape [maxTime, batchSize,
     *      inputSize], otherwise the input has a shape [batchSize, maxTime,
     *      inputSize].
     * * 1: weights.
     *      A 2-D tensor of shape [numUnits, inputSize].
     * * 2: recurrent_weights.
     *      A 2-D tensor of shape [numUnits, numUnits].
     * * 3: bias.
     *      A 1-D tensor of shape [numUnits].
     * * 4: hidden state
     *      A 2-D tensor of shape [batchSize, numUnits]. Specifies a hidden
     *      state input for the first time step of the computation.
     * * 5: fusedActivationFunction.
     *      A {@link FuseCode} value indicating the activation function. If
     *      “NONE” is specified then it results in a linear activation.
     * * 6: timeMajor
     *      An {@link ANEURALNETWORKS_INT32} scalar specifying the shape format
     *      of input and output tensors. Must be set to either 0 or 1.
     * Outputs:
     * * 0: output.
     *      A 3-D tensor. The shape is defined by the input 6 (timeMajor). If
     *      it is set to 1, then the output has a shape [maxTime, batchSize,
     *      numUnits], otherwise the output has a shape [batchSize, maxTime,
     *      numUnits].
     *
     * Available since API level 29.
     */
    ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN = 93,
} OperationCode;
/**
 * Fused activation function types.
 *
 *
 * Available since API level 27.
 */
typedef enum {
    /** NO fused activation function. */
    ANEURALNETWORKS_FUSED_NONE = 0,
    /** Fused ReLU activation function. */
    ANEURALNETWORKS_FUSED_RELU = 1,
    /** Fused ReLU1 activation function. */
    ANEURALNETWORKS_FUSED_RELU1 = 2,
    /** Fused ReLU6 activation function. */
    ANEURALNETWORKS_FUSED_RELU6 = 3,
} FuseCode;
/**
 * Implicit padding algorithms.
 *
 *
 * Available since API level 27.
 */
typedef enum {
    /**
     * SAME padding.
     * Padding on both ends are the "same":
     *     padding_to_beginning =  total_padding / 2
     *     padding_to_end       = (total_padding + 1)/2.
     * i.e., for even number of padding, padding to both ends are exactly
     * the same; for odd number of padding, padding to the ending is bigger
     * than the padding to the beginning by 1.
     *
     * total_padding is a function of input, stride and filter size.
     * It could be computed as follows:
     *    out_size = (input + stride - 1) / stride;
     *    needed_input = (out_size - 1) * stride + filter_size
     *    total_padding = max(0, needed_input - input_size)
     *  The computation is the same for the horizontal and vertical directions.
     */
    ANEURALNETWORKS_PADDING_SAME = 1,
    /**
     * VALID padding.
     * No padding. When the input size is not evenly divisible by
     * the filter size, the input at the end that could not fill
     * the whole filter tile will simply be ignored.
     */
    ANEURALNETWORKS_PADDING_VALID = 2,
} PaddingCode;
/**
 * Execution preferences.
 *
 * Available since API level 27.
 */
typedef enum {
    /**
     * Prefer executing in a way that minimizes battery drain.
     * This is desirable for compilations that will be executed often.
     */
    ANEURALNETWORKS_PREFER_LOW_POWER = 0,
    /**
     * Prefer returning a single answer as fast as possible, even if this causes
     * more power consumption.
     */
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
    /**
     * Prefer maximizing the throughput of successive frames, for example when
     * processing successive frames coming from the camera.
     */
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,
} PreferenceCode;
/**
 * Device types.
 *
 * The type of NNAPI device.
 */
typedef enum {
    /** The device type cannot be provided. */
    ANEURALNETWORKS_DEVICE_UNKNOWN = 0,
    /** The device does not fall into any category below. */
    ANEURALNETWORKS_DEVICE_OTHER = 1,
    /** The device runs NNAPI models on single or multi-core CPU. */
    ANEURALNETWORKS_DEVICE_CPU = 2,
    /** The device can run NNAPI models and also accelerate graphics APIs such
     * as OpenGL ES and Vulkan. */
    ANEURALNETWORKS_DEVICE_GPU = 3,
    /** Dedicated accelerator for Machine Learning workloads. */
    ANEURALNETWORKS_DEVICE_ACCELERATOR = 4,
} DeviceTypeCode;
/**
 * Result codes.
 *
 * <p>Any NNAPI function can return any result code, including result codes not
 * currently documented. Any value other than {@link ANEURALNETWORKS_NO_ERROR}
 * indicates a failure of some kind.</p>
 *
 * <p>Additional information about the nature of a failure can be obtained from
 * the device log after enabling NNAPI debugging by setting the debug.nn.vlog
 * property to 1, e.g., by calling "adb shell setprop debug.nn.vlog 1".</p>
 *
 * Available since API level 27.
 */
typedef enum {
    /**
     * Operation was succesful.
     */
    ANEURALNETWORKS_NO_ERROR = 0,
    /**
     * Failure caused by not enough available memory.
     */
    ANEURALNETWORKS_OUT_OF_MEMORY = 1,
    ANEURALNETWORKS_INCOMPLETE = 2,
    /**
     * Failure caused by unexpected null argument.
     */
    ANEURALNETWORKS_UNEXPECTED_NULL = 3,
    /**
     * Failure caused by invalid function arguments, invalid model definition,
     * invalid execution definition or invalid data at execution time.
     */
    ANEURALNETWORKS_BAD_DATA = 4,
    /**
     * Failure caused by failed model execution.
     */
    ANEURALNETWORKS_OP_FAILED = 5,
    /**
     * Failure caused by object being in the wrong state.
     */
    ANEURALNETWORKS_BAD_STATE = 6,
    /**
     * Failure caused by not being able to map a file into memory.
     * This may be caused by a file descriptor not being mappable, or an
     * AHardwareBuffer not supported by the device. Mitigate by reading its
     * content into memory.
     */
    ANEURALNETWORKS_UNMAPPABLE = 7,
    /**
     * Failure caused by insufficient buffer size provided to a model output.
     */
    ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE = 8,
    /**
     * Failure caused by a device not being available.
     */
    ANEURALNETWORKS_UNAVAILABLE_DEVICE = 9,
} ResultCode;
/**
 * For {@link ANeuralNetworksModel_setOperandValue}, values with a
 * length smaller or equal to this will be immediately copied into
 * the model. The size is in bytes.
 *
 * Available since API level 27.
 */
enum { ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES = 128 };
/**
 * For {@link ANeuralNetworksCompilation_setCaching}, specify the size
 * of the cache token required from the application. The size is in bytes.
 *
 * Available since API level 29.
 */
enum { ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN = 32 };
/**
 * ANeuralNetworksMemory is an opaque type that represents memory.
 *
 * This type is used to represent shared memory, memory mapped files,
 * and similar memories.
 *
 * By using shared memory, a program can efficiently communicate to the
 * runtime and drivers the tensors that define a model. See
 * {@link ANeuralNetworksModel_setOperandValueFromMemory}. An application
 * should typically create one shared memory object that contains every constant
 * tensor needed to define a model. {@link ANeuralNetworksMemory_createFromFd}
 * can be used to create shared memory from a file handle.
 * {@link ANeuralNetworksMemory_createFromAHardwareBuffer} can be used to
 * create shared memory from an AHardwareBuffer handle.
 *
 * Memory objects can also be used to specify the input and output arguments of
 * an execution. See {@link ANeuralNetworksExecution_setInputFromMemory}
 * and {@link ANeuralNetworksExecution_setOutputFromMemory}.
 *
 * Available since API level 27.
 */
typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;
/**
 * ANeuralNetworksModel is an opaque type that contains a description of the
 * mathematical operations that constitute the model.
 *
 * <p>Build the model by calling<ul>
 * <li>{@link ANeuralNetworksModel_create}</li>
 * <li>{@link ANeuralNetworksModel_addOperation}</li>
 * <li>{@link ANeuralNetworksModel_addOperand}</li>
 * </ul>
 *
 * This forms a graph in which each operation and operand is a node, a
 * directed edge from an operand to an operation indicates that the
 * operand is an input to the operation, and a directed edge from an
 * operation to an operand indicates that the operand is an output
 * from the operation. This graph must be acyclic.
 *
 * A model is completed by calling {@link ANeuralNetworksModel_finish}.
 * A model is destroyed by calling {@link ANeuralNetworksModel_free}.
 *
 * <p>A model cannot be modified once {@link ANeuralNetworksModel_finish}
 * has been called on it.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a model at a given time. It is however safe for more than one
 * thread to use the model once {@link ANeuralNetworksModel_finish} has
 * returned.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the model after calling {@link ANeuralNetworksModel_free}. This
 * includes any compilation or execution object created using the model.</p>
 *
 * Available since API level 27.
 */
typedef struct ANeuralNetworksModel ANeuralNetworksModel;
/**
 * ANeuralNetworksCompilation is an opaque type that can be used to compile
 * a machine learning model.
 *
 * <p>To use:<ul>
 *    <li>Create a new compilation instance by calling the
 *        {@link ANeuralNetworksCompilation_create} function or
 *        {@link ANeuralNetworksCompilation_createForDevices}.</li>
 *    <li>Set any desired properties on the compilation (for example,
 *        {@link ANeuralNetworksCompilation_setPreference}).</li>
 *    <li>Optionally, set the caching signature and the cache directory on the
 *        compilation by calling {@link
 * ANeuralNetworksCompilation_setCaching}.</li> <li>Complete the compilation
 * with {@link ANeuralNetworksCompilation_finish}.</li> <li>Use the compilation
 * as many times as needed with {@link ANeuralNetworksExecution_create} and
 *        {@link ANeuralNetworksBurst_create}.</li>
 *    <li>Destroy the compilation with {@link ANeuralNetworksCompilation_free}
 *        once all executions using the compilation have
 * completed.</li></ul></p>
 *
 * A compilation is completed by calling {@link
 * ANeuralNetworksCompilation_finish}. A compilation is destroyed by calling
 * {@link ANeuralNetworksCompilation_free}.
 *
 * <p>A compilation cannot be modified once {@link
 * ANeuralNetworksCompilation_finish} has been called on it.</p>
 *
 * <p>It is the application's responsibility to make sure that only
 * one thread modifies a compilation at a given time. It is however
 * safe for more than one thread to use the compilation once
 * {@link ANeuralNetworksCompilation_finish} has returned.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the compilation after calling {@link
 * ANeuralNetworksCompilation_free}. This includes any execution object created
 * using the compilation.</p>
 *
 * Available since API level 27.
 */
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;
/**
 * ANeuralNetworksExecution is an opaque type that can be used to apply a
 * machine learning model to a set of inputs.
 *
 * <p>To use:<ul>
 *    <li>Create a new execution instance by calling the
 *        {@link ANeuralNetworksExecution_create} function.</li>
 *    <li>Associate input buffers or memory regions to the model inputs with
 *        {@link ANeuralNetworksExecution_setInput} or
 *        {@link ANeuralNetworksExecution_setInputFromMemory}.</li>
 *    <li>Associate output buffers or memory regions to the model outputs with
 *        {@link ANeuralNetworksExecution_setOutput} or
 *        {@link ANeuralNetworksExecution_setOutputFromMemory}.</li>
 *    <li>Apply the model with one of the following:</li><ul>
 *        <li>Asynchronously with {@link ANeuralNetworksExecution_startCompute},
 *            waiting for the execution to complete with
 *            {@link ANeuralNetworksEvent_wait}.</li>
 *        <li>Synchronously with {@link ANeuralNetworksExecution_compute}.</li>
 *        <li>Synchronously as part of an execution burst with
 *            {@link ANeuralNetworksExecution_burstCompute}.</li></ul>
 *    <li>Destroy the execution with
 *        {@link ANeuralNetworksExecution_free}.</li></ul></p>
 *
 * <p>An output buffer or memory region must not overlap with any
 * other output buffer or memory region, with an input buffer or
 * memory region, or with an operand value in a memory object
 * ({@link ANeuralNetworksModel_setOperandValueFromMemory}).</p>
 *
 * <p>An execution cannot be modified once
 * {@link ANeuralNetworksExecution_compute} or
 * {@link ANeuralNetworksExecution_startCompute} has been called on it.</p>
 *
 * <p>An execution can be applied to a model with
 * {@link ANeuralNetworksExecution_compute} or
 * {@link ANeuralNetworksExecution_startCompute} only once. Create new
 * executions to do new evaluations of the model.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies an execution at a given time. It is however safe for more than one
 * thread to use {@link ANeuralNetworksEvent_wait} at the same time.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the execution after calling {@link
 * ANeuralNetworksExecution_free}.</p>
 *
 * <p>Multiple executions can be scheduled and evaluated concurrently, either by
 * means of {@link ANeuralNetworksExecution_compute} (which is synchronous) in
 * different threads or by means of
 * {@link ANeuralNetworksExecution_startCompute} (which is asynchronous). The
 * runtime makes no guarantee on the ordering of completion of executions. If
 * it's important to the application, the application should enforce the
 * ordering by ensuring that one execution completes before the next is
 * scheduled (for example, by scheduling all executions synchronously within a
 * single thread, or by scheduling all executions asynchronously and using
 * {@link ANeuralNetworksEvent_wait} between calls to
 * {@link ANeuralNetworksExecution_startCompute}).</p>
 *
 * Available since API level 27.
 */
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;
#if __ANDROID_API__ >= __ANDROID_API_Q__
/**
 * Parameters for ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL operand.
 */
typedef struct ANeuralNetworksSymmPerChannelQuantParams {
    /* The index of the channel dimension. */
    uint32_t channelDim;
    /** The size of the scale array. Should be equal to dimension[channelDim] of
     * the Operand. */
    uint32_t scaleCount;
    /** The array of scaling values for each channel. Each value must be greater
     * than zero. */
    const float* scales;
} ANeuralNetworksSymmPerChannelQuantParams;
/**
 * ANeuralNetworksBurst is an opaque type that can be used to reduce the latency
 * of a rapid sequence of executions. It will likely cause overhead if only used
 * for a single execution.
 *
 * ANeuralNetworksBurst serves as a context object for any number of inferences
 * using {@link ANeuralNetworksExecution} objects. An ANeuralNetworksBurst
 * object and the {@link ANeuralNetworksExecution} objects used with it must all
 * have been created from the same {@link ANeuralNetworksCompilation} object.
 *
 * This object is also used as a hint to drivers, providing insight to the
 * lifetime of a rapid sequence of executions. For example, a driver may choose
 * to increase the clock frequency of its accelerator for the lifetime of a
 * burst object.
 *
 * <p>To use:<ul>
 *    <li>Create a new burst object by calling the
 *        {@link ANeuralNetworksBurst_create} function.</li>
 *    <li>For each execution:</li><ul>
 *        <li>Create {@link ANeuralNetworksExecution} and configure its
 *            properties (see {@link ANeuralNetworksExecution} for
 * details).</li> <li>Apply the model synchronously with
 *            {@link ANeuralNetworksExecution_burstCompute}, reusing the same
 *            {@link ANeuralNetworksBurst} with the new
 *            {@link ANeuralNetworksExecution}.</li>
 *        <li>Use and free the {@link ANeuralNetworksExecution}.</li></ul>
 *    <li>Destroy the burst with
 *        {@link ANeuralNetworksBurst_free}.</li></ul></p>
 *
 * Available since API level 29.
 */
typedef struct ANeuralNetworksBurst ANeuralNetworksBurst;
#endif  //  __ANDROID_API__ >= __ANDROID_API_Q__
/**
 * ANeuralNetworksOperandType describes the type of an operand.
 * This structure is used to describe both scalars and tensors.
 *
 * A tensor operand type must have a specified rank (number of
 * dimensions) but may have any of its dimensions unspecified.
 *
 * A tensor operand type with all dimensions specified is "fully
 * specified".  Whenever possible (i.e., whenever the dimensions are
 * known at model construction time), a tensor operand type should be
 * (but is not required to be) fully specified, in order to enable the
 * best possible performance.
 *
 * If a tensor operand's type is not fully specified, the dimensions
 * of the operand are deduced from the operand types and values of the
 * operation for which that operand is an output.
 *
 * <p>In the following situations, a tensor operand type must be fully
 * specified:<ul>
 *     <li>The operand has a constant value, set by
 *         {@link ANeuralNetworksModel_setOperandValue} (with a
 *         non-nullptr buffer) or
 *         {@link ANeuralNetworksModel_setOperandValueFromMemory}.</li>
 *     <li>The operand is a model input or model output (see
 *         {@link ANeuralNetworksModel_identifyInputsAndOutputs}).  A
 *         fully specified tensor operand type must either be provided
 *         to {@link ANeuralNetworksModel_addOperand}; or it must be
 *         provided to the corresponding
 *         {@link ANeuralNetworksExecution_setInput},
 *         {@link ANeuralNetworksExecution_setInputFromMemory},
 *         {@link ANeuralNetworksExecution_setOutput}, or
 *         {@link ANeuralNetworksModel_setOperandValueFromMemory}.
 *         EXCEPTION: If the input or output is optional and omitted
 *         (by passing nullptr for buffer to
 *         {@link ANeuralNetworksExecution_setInput} or
 *         {@link ANeuralNetworksExecution_setOutput}) then it need
 *         not have a fully specified tensor operand type.</li></ul>
 *
 * A tensor operand type with some number of unspecified dimensions is
 * represented by setting each unspecified dimension to 0.
 *
 * Available since API level 27.
 */
typedef struct ANeuralNetworksOperandType {
    /** The data type, e.g ANEURALNETWORKS_INT8. */
    int32_t type;
    /** The number of dimensions (rank). It should be 0 for scalars. */
    uint32_t dimensionCount;
    /** The dimensions of the tensor. It should be nullptr for scalars. */
    const uint32_t* dimensions;
    /** These two fields are only used for quantized tensors.
     * They should be zero for scalars and non-fixed point tensors.
     * The dequantized value of each entry is (value - zeroPoint) * scale.
     */
    float scale;
    int32_t zeroPoint;
} ANeuralNetworksOperandType;
typedef int32_t ANeuralNetworksOperationType;
/**
 * ANeuralNetworksEvent is an opaque type that represents an event
 * that will be signaled once an execution completes.
 *
 * Available since API level 27.
 */
typedef struct ANeuralNetworksEvent ANeuralNetworksEvent;
#if __ANDROID_API__ >= __ANDROID_API_Q__
/**
 * ANeuralNetworksDevice is an opaque type that represents a device.
 *
 * This type is used to query basic properties and supported operations of the
 * corresponding device, and control which device(s) a model is to be run on.
 *
 * Available since API level 29.
 */
typedef struct ANeuralNetworksDevice ANeuralNetworksDevice;
/**
 * Get the number of available devices.
 *
 * @param numDevices Used to return the number of devices.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworks_getDeviceCount(uint32_t* numDevices);
/**
 * Get the representation of the specified device.
 *
 * @param devIndex The index of the specified device. Must be less than the
                   number of available devices.
 * @param device The representation of the specified device.
 *               The same representation will always be returned for the
 specified
 *               device.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworks_getDevice(uint32_t devIndex,
                              ANeuralNetworksDevice** device);
/**
 * Get the name of the specified device.
 *
 * @param device The representation of the specified device.
 * @param name   The returned name of the specified device. The name will be in
 * UTF-8 and will be null-terminated. It will be recognizable as a known device
 * name rather than a cryptic string. For devices with feature level 29 and
 * above, the format of the name is {VENDOR}-{DEVICE}, e.g. “google-ipu”. For
 * devices with feature level 28 or lower, the name will always be
 * “unknown-device”. The name will remain valid for the duration of the
 * application.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworksDevice_getName(const ANeuralNetworksDevice* device,
                                  const char** name);
/**
 * Get the type of a given device.
 *
 * The device type can be used to help application developers to distribute
 * Machine Learning workloads and other workloads such as graphical rendering.
 * E.g., for an app which renders AR scenes based on real time object detection
 * results, the developer could choose an ACCELERATOR type device for ML
 * workloads, and reserve GPU for graphical rendering.
 *
 * @param device The representation of the specified device.
 * @param type The returned {@link DeviceTypeCode} of the specified device.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworksDevice_getType(const ANeuralNetworksDevice* device,
                                  int32_t* type);
/**
 * Get the version of the driver implementation of the specified device.
 *
 * It’s the responsibility of the driver implementor to insure that this version
 * string uniquely distinguishes this implementation from all previous
 * implementations.
 *
 * This version string must not be confused with the feature level which is
 * solely defined by {@link ANeuralNetworksDevice_getFeatureLevel}. There is no
 * implicit ordering of the versions. For example, it is not possible to filter
 * all drivers older than a certain version.
 *
 * Application developers may use this version string to avoid or prefer
 * specific driver implementations. For example, an application may want to do
 * so because:
 *     - A specific version of the driver does not provide the required
 * performance, perhaps because of a performance regression.
 *     - A specific version of the driver has a bug or returns results that
 * don’t match the minimum precision requirement for the application.
 *
 * @param device The representation of the specified device.
 * @param version The returned version string of the driver for the specified
 * device. The string will be in UTF-8 and will be null-terminated. For devices
 * with feature level 28 or lower, "UNKOWN" will be returned. The version string
 * will remain valid for the duration of the application.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworksDevice_getVersion(const ANeuralNetworksDevice* device,
                                     const char** version);
/**
 * Get the supported NNAPI version of the specified device.
 *
 * Each device has a supported feature level, which is the most advanced feature
 * this driver implements. For example, if the driver implements the features
 * introduced in Android P, but does not implement the features introduced after
 * Android P, the value would be 28. Developers could decide whether or not the
 * specified device should be used for a Model that has certain feature
 * requirements.
 *
 * @param device The representation of the specified device.
 * @param featureLevel The API level of the most advanced feature this driver
 * implements.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworksDevice_getFeatureLevel(const ANeuralNetworksDevice* device,
                                          int64_t* featureLevel);
/**
 * Get the supported operations for a specified set of devices. If multiple
 * devices are selected, the supported operation list is a union of supported
 * operations of all selected devices.
 *
 * @param model The model to be queried.
 * @param devices The set of devices. Must not contain duplicates.
 * @param numDevices The number of devices in the set.
 * @param supportedOps The boolean array to be filled. True means supported. The
 * size of the boolean array must be at least as large as the number of
 * operations in the model. The order of elements in the supportedOps array
 * matches the order in which the corresponding operations were added to the
 * model.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworksModel_getSupportedOperationsForDevices(
    const ANeuralNetworksModel* model,
    const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
    bool* supportedOps);
/**
 * Create a {@link ANeuralNetworksCompilation} to compile the given model for a
 * specified set of devices. If more than one device is specified, the
 * compilation will distribute the workload automatically across the devices.
 * The model must be fully supported by the specified set of devices. This means
 * that ANeuralNetworksModel_getSupportedOperationsForDevices() must have
 * returned true for every operation for that model/devices pair.
 *
 * @param model The {@link ANeuralNetworksModel} to be compiled.
 * @param devices The set of devices. Must not contain duplicates.
 * @param numDevices The number of devices in the set.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the model is invalid.
 *
 * Available since API level 29.
 */
int ANeuralNetworksCompilation_createForDevices(
    ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
    uint32_t numDevices, ANeuralNetworksCompilation** compilation);
/**
 * Sets the compilation caching signature and the cache directory.
 *
 * Provides optional caching information to the runtime for faster repeated
 * compilation.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be modified.
 * @param cacheDir The cache directory for the runtime to store and retrieve
 * caching data. It is recommended to use the code cache directory provided by
 * the Android runtime. If not using the code cache directory, the user should
 * choose a directory local to the application, and is responsible to managing
 * the cache entries.
 * @param token The token provided by the user to specify a model must be of
 * length ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN. The user should ensure that
 *              the token is unique to a model within the application. The NNAPI
 *              runtime cannot detect token collisions; a collision will result
 * in a failed execution or in a successful execution that produces incorrect
 *              output values.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworksCompilation_setCaching(
    ANeuralNetworksCompilation* compilation, const char* cacheDir,
    const uint8_t* token);
/**
 * Schedule synchronous evaluation of the execution.
 *
 * <p>Schedules synchronous evaluation of the execution. Returns once the
 * execution has completed and the outputs are ready to be consumed.
 * </p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * See {@link ANeuralNetworksExecution_startCompute} for asynchronous execution.
 * Synchronous execution incurs lower overhead than asynchronous execution.
 *
 * Available since API level 29.
 *
 * @param execution The execution to be scheduled and executed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 *         ANEURALNETWORKS_UNMAPPABLE if the execution input or output memory
 * cannot be properly mapped.
 */
int ANeuralNetworksExecution_compute(ANeuralNetworksExecution* execution);
/**
 * Get the dimensional information of the specified output operand of the model
 * of the
 * {@link ANeuralNetworksExecution}.
 *
 * On asynchronous execution initiated by {@link
 * ANeuralNetworksExecution_startCompute},
 * {@link ANeuralNetworksEvent_wait} must be called prior to this function to
 * recuperate the resources used by the execution.
 *
 * @param execution The execution to be queried.
 * @param index The index of the output argument we are querying. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param rank The rank of the output operand.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful,
 * ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE if the target output is provided an
 * insufficient buffer at execution time, ANEURALNETWORKS_BAD_DATA if the index
 * is invalid.
 *
 * Available since API level 29.
 */
int ANeuralNetworksExecution_getOutputOperandRank(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* rank);
/**
 * Get the dimensional information of the specified output operand of the model
 * of the
 * {@link ANeuralNetworksExecution}. The target output operand cannot be a
 * scalar.
 *
 * On asynchronous execution initiated by {@link
 * ANeuralNetworksExecution_startCompute},
 * {@link ANeuralNetworksEvent_wait} must be called prior to this function to
 * recuperate the resources used by the execution.
 *
 * @param execution The execution to be queried.
 * @param index The index of the output argument we are querying. It is an index
 * into the lists passed to {@link
 * ANeuralNetworksModel_identifyInputsAndOutputs}. It is not the index
 * associated with {@link ANeuralNetworksModel_addOperand}.
 * @param dimensions The dimension array to be filled. The size of the array
 * must be exactly as large as the rank of the output operand to be queried in
 * the model.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful,
 * ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE if the target output is provided an
 * insufficient buffer at execution time, ANEURALNETWORKS_BAD_DATA if the index
 * is invalid or if the target is a scalar.
 *
 * Available since API level 29.
 */
int ANeuralNetworksExecution_getOutputOperandDimensions(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* dimensions);
/**
 * Create a {@link ANeuralNetworksBurst} to apply the given compilation.
 * This only creates the burst object. Computation is only performed once
 * {@link ANeuralNetworksExecution_burstCompute} is invoked with a valid
 * {@link ANeuralNetworksExecution} and {@link ANeuralNetworksBurst}.
 *
 * <p>The provided compilation must outlive the burst object.</p>
 *
 * Available since API level 29.
 *
 * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
 * @param burst The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the compilation is invalid.
 */
int ANeuralNetworksBurst_create(ANeuralNetworksCompilation* compilation,
                                ANeuralNetworksBurst** burst)
    __INTRODUCED_IN(29);
/**
 * Destroys the burst object.
 *
 * Available since API level 29.
 *
 * @param burst The burst object to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
void ANeuralNetworksBurst_free(ANeuralNetworksBurst* burst) __INTRODUCED_IN(29);
/**
 * Schedule synchronous evaluation of the execution on a burst object.
 *
 * <p>Schedules synchronous evaluation of the execution. Returns once the
 * execution has completed and the outputs are ready to be consumed.</p>
 *
 * <p>There must be at most one {@link ANeuralNetworksExecution} processing at
 * any given time for any given burst object. Any
 * {@link ANeuralNetworksExecution} launched before the previous has finished
 * will result in ANEURALNETWORKS_BAD_STATE.</p>
 *
 * Available since API level 29.
 *
 * @param burst The burst object to execute on.
 * @param execution The execution to be scheduled and executed. The execution
 *                  must be created from the same {@link
 *                  ANeuralNetworksCompilation} as the burst object.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 */
int ANeuralNetworksExecution_burstCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksBurst* burst)
    __INTRODUCED_IN(29);
/**
 * Creates a shared memory object from an AHardwareBuffer handle.
 *
 * If the shared memory is backed by an AHardwareBuffer of
 * AHARDWAREBUFFER_FORMAT_BLOB format, it can be used the same way as shared
 * memory created from a file handle. See
 * {@link ANeuralNetworksMemory} for a description on how to use this shared
 * memory.
 *
 * If the shared memory is backed by an AHardwareBuffer of a format other than
 * AHARDWAREBUFFER_FORMAT_BLOB, it can only be used for Model inputs and
 * outputs. When calling {@link ANeuralNetworksExecution_setInputFromMemory} or
 * {@link ANeuralNetworksExecution_setOutputFromMemory} with the shared memory,
 * both offset and length must be set to zero and the entire memory region will
 * be associated with the specified input or output operand. There is no
 * guarantee that an arbitrary AHardwareBuffer_Format and
 * AHardwareBuffer_UsageFlags combination can be used by arbitrary devices. The
 * execution will fail if selected set of devices cannot consume the buffer.
 *
 * Calling {@link ANeuralNetworksModel_setOperandValueFromMemory} with shared
 * memory backed by an AHardwareBuffer of a format other than
 * AHARDWAREBUFFER_FORMAT_BLOB is disallowed.
 *
 * TODO(miaowang): add documentation about intended usage with introspection
 * API.
 *
 * Available since API level 29.
 *
 * @param ahwb The AHardwareBuffer handle.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 *
 * @see AHardwareBuffer
 */
int ANeuralNetworksMemory_createFromAHardwareBuffer(
    const AHardwareBuffer* ahwb, ANeuralNetworksMemory** memory);
/**
 * Specifies whether duration of the {@link ANeuralNetworksExecution} is to be
 * measured. By default, duration is not measured.
 *
 * The {@link ANeuralNetworksExecution} must have been created with
 * {@link ANeuralNetworksCompilation_createForDevices} with numDevices = 1.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 29.
 *
 * @param execution The execution to be modified.
 * @param measure 'true' if duration is to be measured, 'false' if not.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksExecution_setMeasureTiming(
    ANeuralNetworksExecution* execution, bool measure);
/**
 * Different duration measurements.
 *
 * Durations are measured in nanoseconds.
 *
 * Available since API level 29.
 */
typedef enum {
    // Execution time on hardware (not driver, which runs on host processor).
    ANEURALNETWORKS_DURATION_ON_HARDWARE = 0,
    // Execution time in driver (including time on hardware).  Excludes overhead
    // such as that of the runtime itself and the IPC needed for the runtime to
    // communicate with the driver.
    ANEURALNETWORKS_DURATION_IN_DRIVER = 1,
} DurationCode;
/**
 * Get the time spent in the specified {@link ANeuralNetworksExecution}, in
 * nanoseconds. The execution must have completed.
 *
 * @param execution The execution to be queried.
 * @param durationCode The measurement to be queried, specified by {@link
 * DurationCode}.
 * @param duration The returned duration. If no measurement was requested by
 *                 {@link ANeuralNetworksExecution_setMeasureTiming}, or for
 * some other reason the duration is not available, UINT64_MAX will be returned.
 *                 A particular device need not support any given measurement.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksExecution_getDuration(
    const ANeuralNetworksExecution* execution, int32_t durationCode,
    uint64_t* duration);
#endif  // __ANDROID_API__ >= __ANDROID_API_Q__
#if __ANDROID_API__ >= 27
/**
 * Creates a shared memory object from a file descriptor.
 *
 * The shared memory is backed by a file descriptor via mmap.
 * See {@link ANeuralNetworksMemory} for a description on how to use
 * this shared memory.
 *
 * Available since API level 27.
 *
 * @param size The requested size in bytes.
 *             Must not be larger than the file size.
 * @param prot The desired memory protection for the mapping.
 *             It is either PROT_NONE or the bitwise OR of one or
 *             more of the following flags: PROT_READ, PROT_WRITE.
 * @param fd The requested file descriptor.
 *           The file descriptor has to be mmap-able. The file
 *           descriptor will be duplicated.
 * @param offset The offset to the beginning of the file of the area to map.
 *               The offset has to be aligned to a page size.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 */
int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd,
                                       size_t offset,
                                       ANeuralNetworksMemory** memory)
    __INTRODUCED_IN(27);
/**
 * Delete a memory object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * Available since API level 27.
 *
 * @param memory The memory object to be freed.
 */
void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory)
    __INTRODUCED_IN(27);
/**
 * Create an empty {@link ANeuralNetworksModel}.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksExecution_compute} or
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * The model should be constructed with calls to
 * {@link ANeuralNetworksModel_addOperation} and
 * {@link ANeuralNetworksModel_addOperand}
 *
 * <p>{@link ANeuralNetworksModel_finish} should be called once the model
 * has been fully constructed.</p>
 *
 * <p>{@link ANeuralNetworksModel_free} should be called once the model
 * is no longer needed.</p>
 *
 * Available since API level 27.
 *
 * @param model The {@link ANeuralNetworksModel} to be created.
 *              Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_create(ANeuralNetworksModel** model)
    __INTRODUCED_IN(27);
/**
 * Destroy a model.
 *
 * The model need not have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param model The model to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
void ANeuralNetworksModel_free(ANeuralNetworksModel* model) __INTRODUCED_IN(27);
/**
 * Indicate that we have finished modifying a model. Required before
 * calling {@link ANeuralNetworksCompilation_create} and
 * {@link ANeuralNetworksCompilation_createForDevices}.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * This function must only be called once for a given model.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param model The model to be finished.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_finish(ANeuralNetworksModel* model)
    __INTRODUCED_IN(27);
/**
 * Add an operand to a model.
 *
 * The order in which the operands are added is important. The first one added
 * to a model will have the index value 0, the second 1, etc. These indexes are
 * used as operand identifiers in
 * {@link ANeuralNetworksModel_addOperation},
 * {@link ANeuralNetworksModel_identifyInputsAndOutputs},
 * {@link ANeuralNetworksModel_setOperandValue},
 * {@link ANeuralNetworksModel_setOperandValueFromMemory},
 * {@link ANeuralNetworksExecution_setInput},
 * {@link ANeuralNetworksExecution_setInputFromMemory},
 * {@link ANeuralNetworksExecution_setOutput},
 * {@link ANeuralNetworksExecution_setOutputFromMemory} and
 * {@link ANeuralNetworksExecution_setOperandValue}.
 *
 * <p>Every operand must be referenced in exactly one of the following
 * ways:<ul>
 *    <li>It is identified as a model input with
 *        {@link ANeuralNetworksModel_identifyInputsAndOutputs}.</li>
 *    <li>It is identified as a constant with
 *        {@link ANeuralNetworksModel_setOperandValue} or
 *        {@link ANeuralNetworksModel_setOperandValueFromMemory}.</li>
 *    <li>It is identified as an output of exactly one operation with
 *        {@link ANeuralNetworksModel_addOperation}.</li></p>
 * <p>An operand that is identified as a model input or as a constant
 * must not also be identified as a model output with
 * {@link ANeuralNetworksModel_identifyInputsAndOutputs}.</p>
 *
 * To build a model that can accommodate inputs of various sizes, as
 * you may want to do for a CNN, leave unspecified the dimensions that
 * will vary at run time.  If you do so, fully specify dimensions
 * when calling {@link ANeuralNetworksExecution_setInput} or
 * {@link ANeuralNetworksExecution_setInputFromMemory}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param model The model to be modified.
 * @param type The {@link ANeuralNetworksOperandType} that describes the shape
 *             of the operand.  Neither the {@link ANeuralNetworksOperandType}
 *             nor the dimensions it points to need to outlive the call to
 *             {@link ANeuralNetworksModel_addOperand}.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type)
    __INTRODUCED_IN(27);
/**
 * Sets an operand to a constant value.
 *
 * Values of length smaller or equal to
 * {@link ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES}
 * are immediately copied into the model.
 *
 * For values of length greater than {@link
 * ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES}, a pointer to the
 * buffer is stored within the model. The application is responsible for not
 * changing the content of this region until all executions using this model
 * have completed. As the data may be copied during processing, modifying the
 * data after this call yields undefined results.
 *
 * For large tensors, using {@link
 * ANeuralNetworksModel_setOperandValueFromMemory} is likely to be more
 * efficient.
 *
 * To indicate that an optional operand should be considered missing,
 * pass nullptr for buffer and 0 for length.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model,
                                         int32_t index, const void* buffer,
                                         size_t length) __INTRODUCED_IN(27);
#if __ANDROID_API__ >= __ANDROID_API_Q__
/**
 * Sets an operand's per channel quantization parameters.
 *
 * Sets parameters required by a tensor of type
 * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}.
 * This function must be called for every tensor of type
 * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} before
 * calling {@link ANeuralNetworksModel_finish}.
 *
 * Available since API level 29.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param channelQuant The per channel quantization parameters for the operand.
 *                    No memory in this struct needs to outlive the call to
 *                    this function.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksSymmPerChannelQuantParams* channelQuant)
    __INTRODUCED_IN(29);
#endif  // __ANDROID_API__ >= __ANDROID_API_Q__
/**
 * Sets an operand to a value stored in a memory object.
 *
 * The content of the memory is not copied. A reference to that memory is stored
 * inside the model. The application is responsible for not changing the content
 * of the memory region until all executions using this model have completed.
 * As the data may be copied during processing, modifying the data after this
 * call yields undefined results.
 *
 * To indicate that an optional operand should be considered missing,
 * use {@link ANeuralNetworksModel_setOperandValue} instead, passing nullptr for
 * buffer.
 *
 * Is disallowed to set an operand value with shared memory backed by an
 * AHardwareBuffer of a format other than AHARDWAREBUFFER_FORMAT_BLOB.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 * See {@link ANeuralNetworksMemory_createFromAHardwarBuffer} for information on
 * AHardwareBuffer usage.
 *
 * Available since API level 27.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_setOperandValueFromMemory(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, size_t offset, size_t length)
    __INTRODUCED_IN(27);
/**
 * Add an operation to a model.
 *
 * @param model The model to be modified.
 * @param type The {@link ANeuralNetworksOperationType} of the operation.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying each operand.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying each operand.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_addOperation(
    ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
    uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
    const uint32_t* outputs) __INTRODUCED_IN(27);
/**
 * Specifies which operands will be the model's inputs and
 * outputs. Every model must have at least one input and one output.
 *
 * An operand cannot be used for both input and output. Doing so will
 * return an error.
 *
 * @param model The model to be modified.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying the input operands.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying the output operands.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 */
int ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs) __INTRODUCED_IN(27);
#if __ANDROID_API__ >= 28
/**
 * Specifies whether {@link ANEURALNETWORKS_TENSOR_FLOAT32} is allowed to be
 * calculated with range and/or precision as low as that of the IEEE 754 16-bit
 * floating-point format. By default, {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 * must be calculated using at least the range and precision of the IEEE 754
 * 32-bit floating-point format.
 *
 * @param model The model to be modified.
 * @param allow 'true' indicates {@link ANEURALNETWORKS_TENSOR_FLOAT32} may be
 *              calculated with range and/or precision as low as that of the
 *              IEEE 754 16-bit floating point format. 'false' indicates
 *              {@link ANEURALNETWORKS_TENSOR_FLOAT32} must be calculated using
 *              at least the range and precision of the IEEE 754 32-bit floating
 *              point format.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * Available since API level 28.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 */
int ANeuralNetworksModel_relaxComputationFloat32toFloat16(
    ANeuralNetworksModel* model, bool allow) __INTRODUCED_IN(28);
#endif  // __ANDROID_API__ >= 28
/**
 * Create a {@link ANeuralNetworksCompilation} to compile the given model.
 *
 * <p>This only creates the object. Compilation is only performed once
 * {@link ANeuralNetworksCompilation_finish} is invoked.</p>
 *
 * <p>{@link ANeuralNetworksCompilation_finish} should be called once
 * all desired properties have been set on the compilation.</p>
 *
 * <p>{@link ANeuralNetworksModel_free} should be called once the compilation
 * is no longer needed.</p>
 *
 * <p>The provided model must outlive the compilation.</p>
 *
 * The model must already have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * Available since API level 27.
 *
 * @param model The {@link ANeuralNetworksModel} to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the model is invalid.
 */
int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation)
    __INTRODUCED_IN(27);
/**
 * Destroy a compilation.
 *
 * The compilation need not have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * Available since API level 27.
 *
 * @param compilation The compilation to be destroyed. Passing NULL is
 * acceptable and results in no operation.
 */
void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation)
    __INTRODUCED_IN(27);
/**
 * Sets the execution preference.
 *
 * <p>Provides guidance to the runtime when trade-offs are possible.</p>
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * Available since API level 27.
 *
 * @param compilation The compilation to be modified.
 * @param preference Either {@link PREFER_LOW_POWER},
 *                  {@link PREFER_SINGLE_FAST_ANSWER}, or
 *                  {@link PREFER_SUSTAINED_SPEED}.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksCompilation_setPreference(
    ANeuralNetworksCompilation* compilation, int32_t preference)
    __INTRODUCED_IN(27);
/**
 * Indicate that we have finished modifying a compilation. Required before
 * calling {@link ANeuralNetworksExecution_create}.
 *
 * An application is responsible to make sure that no other thread uses
 * the compilation at the same time.
 *
 * This function must only be called once for a given compilation.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * Available since API level 27.
 *
 * @param compilation The compilation to be finished.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation)
    __INTRODUCED_IN(27);
/**
 * Create a {@link ANeuralNetworksExecution} to apply the given compilation.
 * This only creates the object. Computation is only performed once
 * {@link ANeuralNetworksExecution_compute} or
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * <p>The provided compilation must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
 * @param execution The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the compilation is invalid.
 */
int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution)
    __INTRODUCED_IN(27);
/**
 * Destroy an execution.
 *
 * <p>If called on an execution for which
 * {@link ANeuralNetworksExecution_startCompute} has been called, the
 * function will return immediately but will mark the execution to be deleted
 * once the computation completes. The related {@link ANeuralNetworksEvent}
 * will be signaled and the {@link ANeuralNetworksEvent_wait} will return
 * ANEURALNETWORKS_ERROR_DELETED.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param execution The execution to be destroyed. Passing NULL is acceptable
 * and results in no operation.
 */
void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution)
    __INTRODUCED_IN(27);
/**
 * Associate a user buffer with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * If the input is optional, you can indicate that it is omitted by
 * passing nullptr for buffer and 0 for length.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with
 *              {@link ANeuralNetworksModel_addOperand}.
 * @param type The {@link ANeuralNetworksOperandType} of the
 *             operand. Unless the input is omitted, this should be
 *             used to specify the dimensions that were left
 *             unspecified when the operand was added to the
 *             model. All other properties of the type must be the
 *             same as specified in the model. If the type is the same
 *             as specified when the model was built, NULL can be
 *             passed. Neither the {@link ANeuralNetworksOperandType}
 *             nor the dimensions it points to need to outlive the call
 *             to {@link ANeuralNetworksExecution_setInput}.
 * @param buffer The buffer containing the data.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution,
                                      int32_t index,
                                      const ANeuralNetworksOperandType* type,
                                      const void* buffer, size_t length)
    __INTRODUCED_IN(27);
/**
 * Associate part of a memory object with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * If the input is optional, you can indicate that it is omitted by
 * using {@link ANeuralNetworks_setInput} instead, passing nullptr for buffer
 * and 0 for length.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 * See {@link ANeuralNetworksMemory_createFromAHardwarBuffer} for information on
 * AHardwareBuffer usage.
 *
 * Available since API level 27.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The {@link ANeuralNetworksOperandType} of the
 *             operand. This should be used to specify the dimensions
 *             that were left unspecified when the operand was added
 *             to the model. All other properties of the type must be
 *             the same as specified in the model. If the type is the
 *             same as specified when the model was built, NULL can be
 *             passed. Neither the {@link ANeuralNetworksOperandType}
 *             nor the dimensions it points to need to outlive the call
 *             to {@link ANeuralNetworksExecution_setInputFromMemory}.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
int ANeuralNetworksExecution_setInputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) __INTRODUCED_IN(27);
/**
 * Associate a user buffer with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * If the output is optional, you can indicate that it is omitted by
 * passing nullptr for buffer and 0 for length.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The {@link ANeuralNetworksOperandType} of the
 *             operand. Unless the output is omitted, this should be
 *             used to specify the dimensions that were left
 *             unspecified when the operand was added to the
 *             model. All other properties of the type must be the
 *             same as specified in the model. If the type is the same
 *             as specified when the model was built, NULL can be
 *             passed. Neither the {@link ANeuralNetworksOperandType}
 *             nor the dimensions it points to need to outlive the call
 *             to {@link ANeuralNetworksExecution_setOutput}.
 * @param buffer The buffer where the data is to be written.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution,
                                       int32_t index,
                                       const ANeuralNetworksOperandType* type,
                                       void* buffer, size_t length)
    __INTRODUCED_IN(27);
/**
 * Associate part of a memory object with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * If the output is optional, you can indicate that it is omitted by
 * using {@link ANeuralNetworks_setOutput} instead, passing nullptr for buffer
 * and 0 for length.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 * See {@link ANeuralNetworksMemory_createFromAHardwarBuffer} for information on
 * AHardwareBuffer usage.
 *
 * Available since API level 27.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The {@link ANeuralNetworksOperandType} of the operand. This
 * should be used to specify the dimensions that were left unspecified when the
 * operand was added to the model. All other properties of the type must be the
 *             same as specified in the model. If the type is the same
 *             as specified when the model was built, NULL can be
 *             passed. Neither the {@link ANeuralNetworksOperandType}
 *             nor the dimensions it points to need to outlive the call
 *             to {@link ANeuralNetworksExecution_setOutputFromMemory}.
 * @param memory The memory where the data is to be stored.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The length in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
int ANeuralNetworksExecution_setOutputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) __INTRODUCED_IN(27);
/**
 * Schedule asynchronous evaluation of the execution.
 *
 * <p>Schedules asynchronous evaluation of the execution. Once the model has
 * been applied and the outputs are ready to be consumed, the returned event
 * will be signaled. Use {@link ANeuralNetworksEvent_wait} to wait for that
 * event.
 * </p>
 *
 * ANeuralNetworksEvent_wait must be called to recuperate the resources used
 * by the execution.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * See {@link ANeuralNetworksExecution_compute} for synchronous execution.
 * Synchronous execution incurs lower overhead than asynchronous execution.
 *
 * Available since API level 27.
 *
 * @param execution The execution to be scheduled and executed.
 * @param event The event that will be signaled on completion. event is set to
 *              NULL if there's an error.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event)
    __INTRODUCED_IN(27);
/**
 * Waits until the execution completes.
 *
 * More than one thread can wait on an event. When the execution completes,
 * all threads will be released.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 27.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 *         ANEURALNETWORKS_UNMAPPABLE if the execution input or output memory
 * cannot be properly mapped.
 */
int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) __INTRODUCED_IN(27);
/**
 * Destroys the event.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 27.
 */
void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) __INTRODUCED_IN(27);
#endif  // __ANDROID_API__ >= 27
__END_DECLS
#endif  // ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_MOCK_H
/** @} */
