#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   build_with_onnx.py
@Time   :   2023/07/27 11:34:10
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   build and learn basic onnx model with onnx api
            refer: https://onnx.ai/onnx/intro/python.html
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from pathlib import Path
import netron
from onnx import TensorProto
from onnx.helper import (
    make_model, make_graph, make_node,
    make_tensor_value_info
)
from onnx.checker import check_model

import numpy as np
from onnx.numpy_helper import from_array, to_array
from onnx import load_tensor_from_string

from onnx import load, helper

from onnxruntime import InferenceSession

from onnx.helper import make_opsetid, make_function

from onnx import AttributeProto
from onnx.helper import make_tensor

import onnx.parser
import onnx.checker
from onnx import shape_inference

from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
import timeit


def netron_web(output_path, port=8081):
    """
    refer: https://github.com/lutzroeder/netron/issues/159
    """
    netron.start(output_path, port)


def base_usage():
    # inputs
    # 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    logger.info(f"The type of X is {type(X)}")

    # outputs, the shape is left undefined
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

    # nodes
    # Create a node defined by the operator MatMul,
    # 'X', 'A' are the inputs of the node, 'XA' the output
    # The output does not difine explicitly
    node1 = make_node("MatMul", ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    logger.info(f"The type of node is {type(node1)}")

    # from node to graph
    # the graph is built from 
    # 1、the list of nodes
    # 2、graph name
    # 3、the list of inputs
    # 4、the list of outputs
    graph = make_graph(
        nodes=[node1, node2],
        name='lr',
        inputs=[X, A, B],
        outputs=[Y]
    )
    logger.info(f"The type of graph is {type(graph)}")

    # onnx model
    # there is no metadata in this case.
    onnx_model = make_model(graph)
    logger.info(f"The type of model is {type(onnx_model)}")

    # Let's check the model is consistent
    check_model(onnx_model)

    # display model
    print(onnx_model)

    # The serialization
    with open("/home/yhao/data/onnx/linear_regression.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


def data_serialization():
    numpy_tensor = np.array([0, 1, 4, 5, 3], dtype=np.float32)
    logger.info(f"The type of numpy_tensor is {type(numpy_tensor)}")

    onnx_tensor = from_array(numpy_tensor)
    # TensorProto
    logger.info(f"The type of onnx_tensor is {type(onnx_tensor)}")

    serialized_tensor = onnx_tensor.SerializeToString()
    # bytes
    logger.info(f"The type of serialized_tensor brefore write is {type(serialized_tensor)}")
    print('serialized_tensor sample: \n', serialized_tensor[:10])

    with open('/home/yhao/data/onnx/saved_tensor.pb', 'wb') as f:
        f.write(serialized_tensor)

    with open('/home/yhao/data/onnx/saved_tensor.pb', 'rb') as f:
        serialized_tensor = f.read()
    logger.info(f"The type of serialized_tensor after read is {type(serialized_tensor)}")

    # total empty init
    onnx_tensor = TensorProto()
    onnx_tensor.ParseFromString(serialized_tensor)
    logger.info(f"The type of onnx_tensor after parse is {type(onnx_tensor)}")

    numpy_tensor = to_array(onnx_tensor)
    print(numpy_tensor)

    # easy version
    with open('/home/yhao/data/onnx/saved_tensor.pb', 'rb') as f:
        proto = load_tensor_from_string(f.read())
    logger.info(f"The type of proto is {type(proto)}")


def initializers(output_path, web=False):
    """Weights and bias should be treated as constant or initializer in the model.
    
    In base_usage weights were treated as inputs of graph, that's not very convenient.
    """
    # initializers
    value = np.array([0.5, -0.6], dtype=np.float32)
    A = from_array(value, name='A')

    value = np.array([0,4], dtype=np.float32)
    C = from_array(value, name='C')

    # dost not change
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    node1 = make_node("MatMul", ['X', 'A'], ['AX'])
    node2 = make_node("Add", ['AX', 'C'], ['Y'])

    graph = make_graph(
        nodes=[node1, node2],
        name='lr',
        inputs=[X],
        outputs=[Y],
        initializer=[A, C]
    )
    onnx_model = make_model(graph)
    check_model(onnx_model)

    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    if web:
        netron_web(str(output_path))


def attributes(output_path, web=False):
    """An attribute is a fixed parameter of an Operator
    """
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

    node_transpose = make_node("Transpose", inputs=["A"], outputs=["tA"], perm=[1,0])
    logger.info(f"The type of node_transpose is {type(node_transpose)}")
    logger.info(f"The type of node_transpose.attribute is {type(node_transpose.attribute)}")
    logger.info(f"The type of node_transpose.attribute[0] is {type(node_transpose.attribute[0])}")
    logger.info(f"The value of AttributeProto.name is {node_transpose.attribute[0].name}")

    node1 = make_node("MatMul", ['X', 'tA'], ['AX'])
    node2 = make_node("Add", ["AX", "B"], ["Y"])

    graph = make_graph(
        nodes=[node_transpose, node1, node2],
        name='lr',
        inputs=[X, A, B],
        outputs=[Y]
    )
    onnx_model = make_model(graph)
    check_model(onnx_model)

    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    if web:
        netron_web(str(output_path))


def opset_and_metadata(input_path):
    with open(input_path, 'rb') as f:
        onnx_model = load(f)

    for field in ['doc_string', 'domain', 'functions',
                  'ir_version', 'metadata_props', 'model_version',
                  'opset_import', 'producer_name', 'producer_version',
                  'training_info']:
        print(field, getattr(onnx_model, field))

    # onnx.onnx_ml_pb2.OperatorSetIdProto
    op_sample = getattr(onnx_model, 'opset_import')[0]
    logger.info(f"The type of optset in opset_import is {type(op_sample)}")
    logger.info(f"The value of domain in OpsetProto is {op_sample.domain}")
    logger.info(f"The value of version in OpsetProto is {op_sample.version}")

    # set some metadata
    onnx_model.model_version = 15
    onnx_model.producer_name = "something"
    onnx_model.producer_version = "some other thing"
    onnx_model.doc_string = "documentation about this model"

    # set metadata_probs, multi items multi model_props
    prop = onnx_model.metadata_props
    logger.info(f"The value of metadata_props is {prop}")
    data = dict(key1="value1", key2="value2")
    helper.set_model_props(onnx_model, data)

    print(onnx_model)


def subgraph_if(output_path, web=False):
    """Such if or for. 
    It is usually better to avoid them as they are not as efficient as the matrix operation are much faster and optimized
    """
    # initializers
    value = np.array([0], dtype=np.float32)
    zero = from_array(value, name="zero")

    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

    # This node build the condition
    rsum = make_node('ReduceSum', ["X"], ["rsum"])
    # compararion
    cond = make_node("Greater", ["rsum", "zero"], ["cond"])

    # Build the graph where the condition is True
    # Input for then
    then_out = make_tensor_value_info("then_out", TensorProto.FLOAT, None)
    # The constant to return
    then_cst = from_array(np.array([1], dtype=np.float32))

    # The only node
    then_const_node = make_node(
        "Constant",
        inputs=[],
        outputs=['then_out'],
        value=then_cst,  # value attribute of the node
        name='cst1'
    )

    # And the graph wrapping these elements.
    then_body = make_graph(
        nodes=[then_const_node],
        name="then_body",
        inputs=[],
        outputs=[then_out]
    )

    # Same process for each branch
    # Why [5]???
    else_out = make_tensor_value_info("else_out", TensorProto.FLOAT, [5])
    else_cst = from_array(np.array([-1], dtype=np.float32))
    else_const_node = make_node(
        "Constant",
        inputs=[],
        outputs=["else_out"],
        value=else_cst,
        name='cst2'
    )
    else_body = make_graph(
        nodes=[else_const_node],
        name="else_body",
        inputs=[],
        outputs=[else_out]
    )

    # Finally the node IF taking both graphs as [attributes]
    if_node = make_node(
        "If",
        inputs=["cond"],
        outputs=["Y"],
        then_branch=then_body,
        else_branch=else_body
    )

    # The final graph
    graph = make_graph(
        nodes=[rsum, cond, if_node],
        name='if',
        inputs=[X],
        outputs=[Y],
        initializer=[zero]
    )
    onnx_model = make_model(graph)
    check_model(onnx_model)

    # Let's freeze the opset
    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ''
    opset.version = 15
    onnx_model.ir_version = 8

    # Save.
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Let's see the output
    sess = InferenceSession(
        path_or_bytes=onnx_model.SerializeToString(),
        providers=['CPUExecutionProvider']
    )

    x = np.ones((3, 2), dtype=np.float32)
    # Signature is important. Same with the graph
    res = sess.run(None, {"X": x})

    # It works.
    print("result", res)
    print()

    # Some display.
    print(onnx_model)


def subgraph_scan(output_path, web=False):
    """Useful to loop over one dimension of a tensor and store the results in a preallocated tensor
    
    A classic nearest neighbours for a regression problem
    """
    # subgraph
    initializers = []
    nodes = []
    inputs = []
    outputs = []

    value = make_tensor_value_info("next_in", 1, [None, 4])
    inputs.append(value)
    value = make_tensor_value_info("next", 1, [None])
    inputs.append(value)

    value = make_tensor_value_info("next_out", 1, [None, None])
    outputs.append(value)
    value = make_tensor_value_info("scan_out", 1, [None])
    outputs.append(value)

    node = make_node(
        'Identity', ['next_in'], ['next_out'],
        name='cdistd_17_Identity', domain='')
    nodes.append(node)

    node = make_node(
        'Sub', ['next_in', 'next'], ['cdistdf_17_C0'],
        name='cdistdf_17_Sub', domain='')
    nodes.append(node)

    node = make_node(
        "ReduceSumSquare", ['cdistdf_17_C0'], ['cdistdf_17_reduced0'],
        name="cdistdf_17_ReduceSumSquare",
        axes=[1],
        keepdims=0,
        domain=''
    )
    nodes.append(node)

    node = make_node(
        "Identity", ['cdistdf_17_reduced0'], ['scan_out'],
        name="cdistdf_17_Identity",
        domain=''
    )
    nodes.append(node)

    graph = make_graph(
        nodes=nodes,
        name="OnnxIdentity",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers
    )

    # main graph
    initializers = []
    nodes = []
    inputs = []
    outputs = []

    opsets = {"": 15, "ai.onnx.ml": 15}
    target_opset = 15  # subgraph

    # initializers
    list_value = [23.29599822460675, -120.86516699239603, -144.70495899914215, -260.08772982740413,
                154.65272105889147, -122.23295157108991, 247.45232560871727, -182.83789715805776,
                -132.92727431421793, 147.48710175784703, 88.27761768038069, -14.87785569894749,
                111.71487894705504, 301.0518319089629, -29.64235742280055, -113.78493504731911,
                -204.41218591022718, 112.26561056133608, 66.04032954135549,
                -229.5428380626701, -33.549262642481615, -140.95737409864623, -87.8145187836131,
                -90.61397011283958, 57.185488100413366, 56.864151796743855, 77.09054590340892,
                -187.72501631246712, -42.779503579806025, -21.642642730674076, -44.58517761667535,
                78.56025104939847, -23.92423223842056, 234.9166231927213, -73.73512816431007,
                -10.150864499514297, -70.37105466673813, 65.5755688281476, 108.68676290979731, -78.36748960443065]
    value = np.array(list_value, dtype=np.float64).reshape((2, 20))
    tensor = from_array(
        value, name='knny_ArrayFeatureExtractorcst')
    initializers.append(tensor)

    list_value = [1.1394007205963135, -0.6848101019859314, -1.234825849533081, 0.4023416340351105,
                0.17742614448070526, 0.46278226375579834, -0.4017809331417084, -1.630198359489441,
                -0.5096521973609924, 0.7774903774261475, -0.4380742907524109, -1.2527953386306763,
                -1.0485529899597168, 1.950775384902954, -1.420017957687378, -1.7062702178955078,
                1.8675580024719238, -0.15135720372200012, -0.9772778749465942, 0.9500884413719177,
                -2.5529897212982178, -0.7421650290489197, 0.653618574142456, 0.8644362092018127,
                1.5327792167663574, 0.37816253304481506, 1.4693588018417358, 0.154947429895401,
                -0.6724604368209839, -1.7262825965881348, -0.35955315828323364, -0.8131462931632996,
                -0.8707971572875977, 0.056165341287851334, -0.5788496732711792, -0.3115525245666504,
                1.2302906513214111, -0.302302747964859, 1.202379822731018, -0.38732680678367615,
                2.269754648208618, -0.18718385696411133, -1.4543657302856445, 0.04575851559638977,
                -0.9072983860969543, 0.12898291647434235, 0.05194539576768875, 0.7290905714035034,
                1.4940791130065918, -0.8540957570075989, -0.2051582634449005, 0.3130677044391632,
                1.764052391052246, 2.2408931255340576, 0.40015721321105957, 0.978738009929657,
                0.06651721894741058, -0.3627411723136902, 0.30247190594673157, -0.6343221068382263,
                -0.5108051300048828, 0.4283318817615509, -1.18063223361969, -0.02818222902715206,
                -1.6138978004455566, 0.38690251111984253, -0.21274028718471527, -0.8954665660858154,
                0.7610377073287964, 0.3336743414402008, 0.12167501449584961, 0.44386324286460876,
                -0.10321885347366333, 1.4542734622955322, 0.4105985164642334, 0.14404356479644775,
                -0.8877857327461243, 0.15634897351264954, -1.980796456336975, -0.34791216254234314]
    value = np.array(list_value, dtype=np.float32).reshape((20, 4))
    tensor = from_array(value, name='Sc_Scancst')
    initializers.append(tensor)

    value = np.array([2], dtype=np.int64)
    tensor = from_array(value, name='To_TopKcst')
    initializers.append(tensor)

    value = np.array([2, -1, 2], dtype=np.int64)
    tensor = from_array(value, name='knny_Reshapecst')
    initializers.append(tensor)

    # inputs
    value = make_tensor_value_info("input", 1, [None, 4])
    inputs.append(value)

    # outputs
    value = make_tensor_value_info("variable", 1, [None, 2])
    outputs.append(value)

    # nodes
    node = make_node(
        "Scan",
        inputs=["input", "Sc_Scancst"],
        outputs=["UU032UU", "UU033UU"],
        name="Sc_Scan",
        body=graph,
        num_scan_inputs=1,  # TODO Scan and num_scan_inputs
        domain=""
    )
    nodes.append(node)

    node = make_node(
        'Transpose', ['UU033UU'], ['Tr_transposed0'],
        name='Tr_Transpose', perm=[1, 0], domain='')
    nodes.append(node)

    node = make_node(
        'Sqrt', ['Tr_transposed0'], ['Sq_Y0'],
        name='Sq_Sqrt', domain='')
    nodes.append(node)
    
    node = make_node(
        'TopK', ['Sq_Y0', 'To_TopKcst'], ['To_Values0', 'To_Indices1'],
        name='To_TopK', largest=0, sorted=1, domain='')
    nodes.append(node)

    node = make_node(
        'Flatten', ['To_Indices1'], ['knny_output0'],
        name='knny_Flatten', domain='')
    nodes.append(node)

    # TODO what's this?
    node = make_node(
        'ArrayFeatureExtractor',
        ['knny_ArrayFeatureExtractorcst', 'knny_output0'], ['knny_Z0'],
        name='knny_ArrayFeatureExtractor', domain='ai.onnx.ml')
    nodes.append(node)

    # TODO reshape two inputs?
    node = make_node(
        'Reshape', ['knny_Z0', 'knny_Reshapecst'], ['knny_reshaped0'],
        name='knny_Reshape', allowzero=0, domain='')
    nodes.append(node)

    node = make_node(
        'Transpose', ['knny_reshaped0'], ['knny_transposed0'],
        name='knny_Transpose', perm=[1, 0, 2], domain='')
    nodes.append(node)

    node = make_node(
        'Cast', ['knny_transposed0'], ['Ca_output0'],
        name='Ca_Cast', to=TensorProto.FLOAT, domain='')
    nodes.append(node)

    node = make_node(
        'ReduceMean', ['Ca_output0'], ['variable'],
        name='Re_ReduceMean', axes=[2], keepdims=0, domain='')
    nodes.append(node)

    # graph
    graph = make_graph(nodes, 'KNN regressor', inputs, outputs, initializers)

    # model
    onnx_model = make_model(graph)
    onnx_model.ir_version = 8
    onnx_model.producer_name = 'skl2onnx'
    onnx_model.producer_version = ''
    onnx_model.domain = 'ai.onnx'
    onnx_model.model_version = 0
    onnx_model.doc_string = ''
    helper.set_model_props(onnx_model, {})

    # opsets
    del onnx_model.opset_import[:]  # pylint: disable=E1101  # TODO ??
    for dom, value in opsets.items():
        op_set = onnx_model.opset_import.add()
        op_set.domain = dom
        op_set.version = value

    check_model(onnx_model)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(onnx_model)

    if web:
        netron_web(str(output_path))


def custom_func_woattr(output_path, web=False):
    """It works like a graph with less types
    shorten the code to build the model and 
    offer more possibilities to the runtime running predictions to be faster 
    if there exists a specific implementation of this function
    """

    new_domain = "custom"
    opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]
    
    # Let's define a function for a linear regression
    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])

    linear_regression = make_function(
        domain=new_domain,
        fname='LinearRegression',
        inputs=["X", "A", "B"],
        outputs=["Y"],
        nodes=[node1, node2],
        opset_imports=opset_imports,
        attributes=[]
    )

    # Let's use it in a graph
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

    graph = make_graph(
        nodes=[
            make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
            make_node("Abs", ["Y1"], ["Y"])
        ],
        name="example",
        inputs=[X, A, B],
        outputs=[Y],
    )

    onnx_model = make_model(
        graph=graph,
        opset_imports=opset_imports,
        functions=[linear_regression]  # functions to add
    )

    check_model(onnx_model)
    print(onnx_model)

    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())


def custom_func_wiattr(output_path, web=False):
    """Netron 7.1 cannot read properties of null (reading 'name')
    """
    new_domain = 'custom'
    opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

    # Let's define a function for a linear regression
    # The first step consists in creating a constant equal to the input parameter of the function
    cst = make_node("Constant", [], ["B"])

    att = AttributeProto()
    att.name = "value"

    # Tihs line indicates the value comes from the argument named 'bias' the function is given
    att.ref_attr_name = "bias"
    att.type = AttributeProto.TENSOR
    cst.attribute.append(att)

    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])

    linear_regression = make_function(
        domain=new_domain,
        fname='LinearRegression',
        inputs=["X", "A"],          # without input B
        outputs=["Y"],
        nodes=[cst, node1, node2],  # input --> node
        opset_imports=opset_imports,
        attributes=["bias"]         # attribute name list
    )

    # Let's use it in a graph.
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

    graph = make_graph(
        nodes=[
            make_node("LinearRegression", ["X", "A"], ["Y1"], domain=new_domain,
                      # Now bias is an argument of the function and is defined as a tensor
                      bias=make_tensor(name="former_B", data_type=TensorProto.FLOAT, dims=[1], vals=[0.67])),
            make_node('Abs', ['Y1'], ['Y']),
        ],
        name='example',
        inputs=[X, A],
        outputs=[Y]
    )

    onnx_model = make_model(
        graph, opset_imports=opset_imports,
        functions=[linear_regression]  # functions to add
    )

    check_model(onnx_model)
    print(onnx_model)

    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())


def checker_and_shape_inference():
    """
    check_model:
        This work for all operators defined in the main domain or the ML domain. 
        It remains silent for any custom operator not defined in any specification

    shape_inference:
        estimate the shape and the type of intermediate results. 
        If known, the runtime can estimate the memory consumption beforehand and optimize the computation. 
        It can fuse some operators, it can do the computation inplace…

        Shape inference does not work all the time. For example, a Reshape operator. 
        Shape inference only works if the shape is constant. 
        If not constant, the shape cannot be easily inferred unless the following nodes expect specific shape
    """
    input = '''
        <
            ir_version: 8,
            opset_import: [ "" : 15]
        >
        agraph (float[I,4] X, float[4,2] A, int[4] B) => (float[I] Y) {
            XA = MatMul(X, A)
            Y = Add(XA, B)
        }
        '''
    try:
        onnx_model = onnx.parser.parse_model(input)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(e)

    input = '''
        <
            ir_version: 8,
            opset_import: [ "" : 15]
        >
        agraph (float[I,4] X, float[4,2] A, float[4] B) => (float[I] Y) {
            XA = MatMul(X, A)
            Y = Add(XA, B)
        }
        '''
    onnx_model = onnx.parser.parse_model(input)
    inferred_model = shape_inference.infer_shapes(onnx_model)
    logger.info(f"type inferred_model: {type(inferred_model)}")  # ModelProto
    print(inferred_model)


def evaluation_and_runtime():
    """runtime.InferenceSession: subgraph_if
    
    Similar code would also work on GraphProto or FunctionProto.
    """
    # Evaluation of a model
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
    onnx_model = make_model(graph)
    check_model(onnx_model)

    sess = ReferenceEvaluator(onnx_model)

    x = np.random.randn(4, 2).astype(np.float32)
    a = np.random.randn(2, 1).astype(np.float32)
    b = np.random.randn(1, 1).astype(np.float32)
    feeds = {'X': x, 'A': a, 'B': b}

    print(sess.run(None, feeds))

    # Evaluation of a node
    node = make_node("EyeLike", ['X'], ['Y'])
    sess = ReferenceEvaluator(node)
    x = np.random.randn(4, 2).astype(np.float32)
    feeds = {'X': x}

    print(sess.run(None, feeds))


def evaluation_step_by_step():
    """
    Complex models usually do not work on the first try and seeing intermediate results may help to find the part incorrectly converted. 
    Parameter verbose displays information about intermediate results.
    """
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
    onnx_model = make_model(graph)
    check_model(onnx_model)

    for verbose in [1, 2, 3, 4]:
        print()
        print(f"------ verbose={verbose}")
        print()
        sess = ReferenceEvaluator(onnx_model, verbose=verbose)

        x = np.random.randn(4, 2).astype(np.float32)
        a = np.random.randn(2, 1).astype(np.float32)
        b = np.random.randn(1, 1).astype(np.float32)
        feeds = {'X': x, 'A': a, 'B': b}

        print(sess.run(None, feeds))


def evaluate_custom_node(output_dir):
    """
    combine operators EyeLike and Add into AddEyeLike to make it more efficient. 
    Next example replaces these two operators by a single one from domain 'optimized'.

    fusion:
        Two consecutive operators are fused into an optimized version of both
    """
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    node0 = make_node('EyeLike', ['A'], ['Eye'])
    node1 = make_node('Add', ['A', 'Eye'], ['A1'])
    node2 = make_node('MatMul', ['X', 'A1'], ['XA1'])
    node3 = make_node('Add', ['XA1', 'B'], ['Y'])
    graph_orin = make_graph([node0, node1, node2, node3], 'lr_orin', [X, A, B], [Y])
    onnx_model_orin = make_model(graph_orin)
    check_model(onnx_model_orin)
    with open(output_dir / "linear_regression_orin.onnx", "wb") as f:
        f.write(onnx_model_orin.SerializeToString())
    sess_orin = ReferenceEvaluator(onnx_model_orin, verbose=2)

    node01 = make_node("AddEyeLike", ['A'], ['A1'], domain="optimized")
    graph_cust = make_graph([node01, node2, node3], 'lr_cust', [X, A, B], [Y])
    onnx_model_cust = make_model(graph_cust, opset_imports=[
        make_opsetid('', 18), make_opsetid('optimized', 1)
    ])
    check_model(onnx_model_cust)
    with open(output_dir / "linear_regression_cust.onnx", "wb") as f:
        f.write(onnx_model_cust.SerializeToString())

    # define the oprun of custom node for the purpose of inference
    class AddEyeLike(OpRun):

        op_domain = "optimized"

        def _run(self, X, alpha=1.0):
            assert len(X.shape) == 2
            assert X.shape[0] == X.shape[1]
            X = X.copy()
            ind = np.diag_indices(X.shape[0])
            X[ind] += alpha
            return (X, )

    sess_cust = ReferenceEvaluator(
        proto=onnx_model_cust,
        verbose=2,
        new_ops=[AddEyeLike]
    )

    # Let's check with the previous model.
    x = np.random.randn(4, 2).astype(np.float32)
    a = np.random.randn(2, 2).astype(np.float32) / 10
    b = np.random.randn(1, 2).astype(np.float32)
    feeds = {'X': x, 'A': a, 'B': b}

    logger.info(f"Reference progress of onnx_model_orin")
    y_orin = sess_orin.run(None, feeds)[0]
    logger.info(f"Reference progress of onnx_model_cust")
    y_cust = sess_cust.run(None, feeds)[0]
    print(y_orin)
    print(y_cust)
    print(f"difference: {np.abs(y_orin - y_cust).max()}")

    # Efficency comparasion
    x = np.random.randn(4, 1000).astype(np.float32)
    a = np.random.randn(1000, 1000).astype(np.float32) / 10
    b = np.random.randn(1, 1000).astype(np.float32)
    feeds = {'X': x, 'A': a, 'B': b}
    
    # proto: str
    sess_orin = ReferenceEvaluator(str(output_dir / "linear_regression_orin.onnx"))
    sess_cust = ReferenceEvaluator(str(output_dir / "linear_regression_cust.onnx"), new_ops=[AddEyeLike])
    y_orin = sess_orin.run(None, feeds)[0]
    y_cust = sess_cust.run(None, feeds)[0]
    print(f"difference: {np.abs(y_orin - y_cust).max()}")
    print(f"time with EyeLike+Add: {timeit.timeit(lambda: sess_orin.run(None, feeds), number=1000)}")
    print(f"time with AddEyeLike: {timeit.timeit(lambda: sess_cust.run(None, feeds), number=1000)}")


def implementation_details():
    """
    Python and C++
        onnx relies on protobuf to define its type. 
        You would assume that a python object is just a wrapper around a C pointer on the internal structure. 
        Therefore, it should be possible to access internal data from a function receiving a python object of type ModelProto. 
        But it is not. According to Protobuf 4, changes, this is no longer possible after version 4 and it is safer to 
        assume the only way to get a hold on the content is to serialize the model into bytes, give it the C function, 
        then deserialize it. Functions like check_model or shape_inference are calling SerializeToString then ParseFromString 
        before checking the model with a C code. 
    Attributes and inputs
        There is a clear distinction between the two. 
        Inputs are dynamic and may change at every execution. 
        Attributes never changes and an optimizer can improve the execution graph assuming it never changes. 
        Therefore, it is impossible to turn an input into an attribute. 
        And the operator Constant is the only operator changing an attribute into an input.
    """
    pass


def shape_or_no_shape():
    """
    onnx usually expects a shape for every input or output assuming the rank (or the number of dimensions) is known. 
    What if we need to create a valid graph for every dimension? This case is still puzzling.
    """
    def create_model(shapes):
        new_domain = 'custom'
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'A'], ['Y'])

        X = make_tensor_value_info('X', TensorProto.FLOAT, shapes['X'])
        A = make_tensor_value_info('A', TensorProto.FLOAT, shapes['A'])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, shapes['Y'])

        graph = make_graph([node1, node2], 'example', [X, A], [Y])

        onnx_model = make_model(graph, opset_imports=opset_imports)
        # Let models runnable by onnxruntime with a released ir_version
        onnx_model.ir_version = 8

        return onnx_model

    print("----------- case 1: 2D x 2D -> 2D")
    onnx_model = create_model({'X': [None, None], 'A': [None, None], 'Y': [None, None]})
    check_model(onnx_model)
    sess = InferenceSession(onnx_model.SerializeToString(),
                            providers=["CPUExecutionProvider"])
    res = sess.run(None, {
        'X': np.random.randn(2, 2).astype(np.float32),
        'A': np.random.randn(2, 2).astype(np.float32)})
    print(res)

    print("----------- case 2: 2D x 1D -> 1D")
    onnx_model = create_model({'X': [None, None], 'A': [None], 'Y': [None]})
    check_model(onnx_model)
    sess = InferenceSession(onnx_model.SerializeToString(),
                            providers=["CPUExecutionProvider"])
    res = sess.run(None, {
        'X': np.random.randn(2, 2).astype(np.float32),
        'A': np.random.randn(2).astype(np.float32)})
    print(res)

    print("----------- case 3: 2D x 0D -> 0D")
    onnx_model = create_model({'X': [None, None], 'A': [], 'Y': []})
    check_model(onnx_model)
    try:
        InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])
    except Exception as e:
        print(e)

    print("----------- case 4: 2D x None -> None")
    onnx_model = create_model({'X': [None, None], 'A': None, 'Y': None})
    try:
        check_model(onnx_model)
    except Exception as e:
        print(type(e), e)
    sess = InferenceSession(onnx_model.SerializeToString(),
                            providers=["CPUExecutionProvider"])
    res = sess.run(None, {
        'X': np.random.randn(2, 2).astype(np.float32),
        'A': np.random.randn(2).astype(np.float32)})
    print(res)
    print("----------- end")


if __name__ == "__main__":

    output_path = Path("/home/yhao/data/onnx")

    # base_usage()
    # data_serialization()
    # initializers(output_path / 'lr_with_init.onnx', web=True)
    # attributes(output_path / 'lr_with_attributes.onnx')
    # opset_and_metadata(output_path / "linear_regression.onnx")
    # subgraph_if(output_path / "onnx_if_sign.onnx")
    # subgraph_scan(output_path / "knnr.onnx", web=False)
    # custom_func_woattr(output_path / "custom_func_woattr.onnx", web=False)
    # custom_func_wiattr(output_path / "custom_func_wiattr.onnx", web=False)
    # checker_and_shape_inference()
    # evaluation_and_runtime()
    # evaluation_step_by_step()
    # evaluate_custom_node(output_path)
    shape_or_no_shape()

    print()