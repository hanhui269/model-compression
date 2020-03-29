import logging
import re
import torch
from collections import OrderedDict

def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module


class GNode:
    """
    It is used to represent a node in model graph, in this graph a module is a node,
    a function out of module (in ```forward``` function) could also be a node.
    """

    def __init__(self, node_name, node_type, op_type, inputs, outputs, nodes):
        """
        Parameters
        ----------
        node_name : str
            It is module name if the node is a module, it is ```scope_name.node_kind.seq``` if it is a func
        node_type : str
            It only has two options: `module` or `func`
        op_type : str
            The operation type of the module or func
        inputs : list of str
            All the inputs of this node, each element is debugName of one input
        outputs : list of str
            All the outputs of this node, each element is debugName of one output
        nodes : list of node
            All the trace graph nodes included in this module or func
        """
        self.name = node_name
        self.type = node_type
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        # store supplementary information for different op types
        # for example, for ```view``` it stores the shape of its input and output
        self.auxiliary = None

class AtenNode:
    """
    It is used to represent a node in model graph, in this graph a module is a node,
    a function out of module (in ```forward``` function) could also be a node.
    """

    def __init__(self, node_name, op_type, inputs, outputs):
        """
        Parameters
        ----------
        node_name : str
            It is module name if the node is a module, it is ```scope_name.node_kind.seq``` if it is a func
        node_type : str
            It only has two options: `module` or `func`
        op_type : str
            The operation type of the module or func
        inputs : list of str
            All the inputs of this node, each element is debugName of one input
        outputs : list of str
            All the outputs of this node, each element is debugName of one output
        nodes : list of node
            All the trace graph nodes included in this module or func
        """
        self.name = node_name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        # store supplementary information for different op types
        # for example, for ```view``` it stores the shape of its input and output
        self.auxiliary = None


class WNode:
    """
    It is used to represent a node in model graph, in this graph a module is a node,
    a function out of module (in ```forward``` function) could also be a node.
    """

    def __init__(self, node_type, module_weight_name, inputs, outputs,tensor_type):
        self.module_weight_name=module_weight_name
        self.ClassType = node_type
        self.inputs = inputs
        self.outputs = outputs
        self.tensor_type = tensor_type

        # store supplementary information for different op types
        # for example, for ```view``` it stores the shape of its input and output
        self.auxiliary = None



class AutoGraph(object):
    """
    This class is to speedup the model with provided weight mask
    """

    def __init__(self, trace):
        """
        Parameters
        ----------
        model : pytorch model
            The model user wants to speed up
        dummy_input : pytorch tensor
            The dummy input for ```jit.trace```, users should put it on right device before pass in
        masks_file : str
            The path of user provided mask file
       """

        self.trace_graph = trace

        self.inferred_masks = dict()  # key: module_name, value: ModuleMasks
        self.g_nodes = list()
        self.global_count = 0
        self._build_graph()

    def up_to_self(self,current_output):
        cur_node=self.weight_nodes[current_output]
        if cur_node.inputs=='self':
            self.path2self_weight_name.insert(0,cur_node.module_weight_name)
            self.classTypes.insert(0,cur_node.ClassType)
            return self.path2self_weight_name
        else:
            self.path2self_weight_name.insert(0,cur_node.module_weight_name)
            self.classTypes.insert(0,cur_node.ClassType)

            self.up_to_self(cur_node.inputs)

    def wrap_WNode(self,w_node):
        """

        :param w_node:  w_node  class
        :return:
        """
        for cur_out,value in self.weight_nodes.items():
            cur_input=value.inputs    #module_weight_name, inputs, outputs,tensor_type
            self.path2self_weight_name = []
            self.classTypes=[]
            self.up_to_self(cur_out)
            print("------Check the matched weight names------>>:",'.'.join(self.path2self_weight_name))

            self.weight_nodes[cur_out].auxiliary='.'.join(self.path2self_weight_name)
            if self.weight_nodes[cur_out].ClassType=='':
                self.weight_nodes[cur_out].ClassType = '.'.join(self.classTypes)

    #
    def wrap_atten_ops(self):
        weight_nodes_keys = self.weight_nodes.keys()
        atten_ops_kesy = self.atten_ops.keys()

        for key, node_values in self.atten_ops.items():
            inputs = node_values.inputs
            forward_layers = []
            behind_layers = []
            up_to_inputs = []
            for input in inputs:
                assert isinstance(input, str), "input must be string"

                if input == 'input.1':
                    forward_layers.append('Data')




                else:
                    if input in weight_nodes_keys:      #todo spite these weights that are bn weights.

                        if "BatchNorm" in self.weight_nodes[input].ClassType or  "bn" in self.weight_nodes[input].ClassType:    #weight
                            forward_layers.append(self.weight_nodes[input].auxiliary)
                        elif "Linear" in self.weight_nodes[input].ClassType and ('bias' in self.weight_nodes[input].module_weight_name or 'weight' in self.weight_nodes[input].module_weight_name):   #weight connect fc
                            forward_layers.append(self.weight_nodes[input].auxiliary)
                            if 'weight' in self.weight_nodes[input].module_weight_name:
                                behind_layers.append(self.weight_nodes[input].auxiliary)

                        else:

                            behind_layers.append(self.weight_nodes[input].auxiliary)
                    elif input in atten_ops_kesy:
                        up_to_inputs.append(input)

            while len(up_to_inputs) >= 1:
                assert isinstance(up_to_inputs[0], str), "up_to_inputs[0] must be string"

                tmp_inputs = self.atten_ops[up_to_inputs[0]].inputs
                has_weight = []
                has_ops = []
                for tmp_input in tmp_inputs:
                    if tmp_input in weight_nodes_keys:
                        has_weight.append(tmp_input)
                    elif tmp_input in atten_ops_kesy:
                        has_ops.append(tmp_input)

                weight_bns_flag=True
                if bool(has_weight):  #逻辑：当遇到weight时候，则进行终止
                    for one in has_weight:
                        forward_layers.append(self.weight_nodes[one].auxiliary)
                        if "BatchNorm" in self.weight_nodes[one].ClassType or  "bn" in self.weight_nodes[one].ClassType:
                            weight_bns_flag*=True

                        # elif "inear" in self.weight_nodes[one].ClassType and ('bias' in self.weight_nodes[one].module_weight_name or 'weight' in self.weight_nodes[one].module_weight_name):   #weight connect fc
                        #     weight_bns_flag*=True

                        else:
                            weight_bns_flag*=False

                if weight_bns_flag:  # 全是bn weight
                    for o in has_ops:
                        up_to_inputs.append(o)


                up_to_inputs.remove(up_to_inputs[0])

            self.atten_ops[key].auxiliary = {"forward": forward_layers, "behind": behind_layers}

    def _build_graph(self):
        """
        Build graph using our defined format from jit trace.

        """
        graph = self.trace_graph.graph
        output_to_node = dict()
        input_to_node = dict()
        weight_nodes=OrderedDict()
        atten_ops=OrderedDict()
        graph_inputs = list()
        graph_outputs = list()
        for _input in graph.inputs():
            graph_inputs.append(_input.debugName())
        for output in graph.outputs():
            graph_outputs.append(output.debugName())
        """
        torch._C.Node         object has no attribute '        type        module_weight_name
        torch/C/Node.py    nodes
        """
        for node in graph.nodes():
            # populate output_to_node and input_to_node
            for output in node.outputs():
                output_name = output.debugName()
                output_to_node[output_name] = node
            for _input in node.inputs():
                input_name = _input.debugName()
                input_to_node[input_name] = node
            scope_name = node.scopeName()  # example: scope_name, 'MyCell/Linear[linear]'
            # if module_name is empty, it is not a module
            # if True:
            if scope_name == '':
                """
                this function is to construct the relation between weight name and jit.trace
                """
                _outputs_name_cur=''
                _input_name_cur=''
                for output in node.outputs():
                    _outputs_name_cur=output.debugName()
                for _input in node.inputs():
                    _input_name_cur=_input.debugName()

                module_weight_name = re.findall(r'\"(.*?)\"',re.findall(r'\[(.*?)\]', str(node))[0])[0]
                ClassType = ''
                tensor=''
                if 'Tensor'in str(node):
                    assert 'ClassType' not in str(node), 'ClassType and tensor must not appear at the same time.'
                    tensor_type = re.findall(r'\"(.*?)\"',re.findall(r'\[(.*?)\]', str(node))[0])[0]
                    tensor=tensor_type
                elif 'ClassType'in str(node):
                    assert 'tensor' not in str(node), 'ClassType and tensor must not appear at the same time.'

                    ClassType=re.findall(r'\<(.*?)\>', str(node))[0]

                weight_node=WNode(node_type=ClassType, module_weight_name=module_weight_name,inputs=_input_name_cur,
                                  outputs=_outputs_name_cur,tensor_type=tensor)
                weight_nodes[_outputs_name_cur] = weight_node

            else:
                str_node = str(node)
                """
                this function is to construct the connection of  jit.trace.
                """
                if node.kind().startswith('prim::'):
                    pass

                else:



                    inputs_name=[]
                    _outputs_names=''
                    op_type=''
                    # inputs=node.inputs()
                    # out_name=node.inputs()
                    for _input in node.inputs():
                        inputs_name.append(_input.debugName())
                    for output in node.outputs():
                        _outputs_names = output.debugName()
                    op_type=node.kind()
                    # str_node.strip().split('#')[0]
                    atten_ops[_outputs_names]=AtenNode(node_name=_outputs_names, op_type=op_type,
                                                       inputs=inputs_name, outputs=_outputs_names)

        self.weight_nodes = weight_nodes     #weight_nodes转化成class，todo：may be used in others
        self.atten_ops=atten_ops             #atten_ops，todo：may be used in others
        self.wrap_WNode(self.weight_nodes)   #add auxilary: weight name
        self.wrap_atten_ops()               #add auxilary: ops realationship name


    def __call__(self,current_weight):
        print('------The curret weight name is :------>>>>>>',current_weight)


        assert isinstance(current_weight, str), "current_weight name must be string"
        if_first_layer=False
        for key, value_node in self.atten_ops.items():
            behind = value_node.auxiliary['behind']
            forwards = value_node.auxiliary['forward']
            if "Data" in forwards and current_weight in behind:
                if_first_layer=True
        if if_first_layer:
            """
                           the first layer, 找出下一层连接的层的forward的共同元素。
            """
            print('*****First layer******', current_weight)

            behind_all_layers = []
            forwards__all_layers = []
            assert isinstance(current_weight, str), "current_weight name must be string"

            for key, value_node in self.atten_ops.items():
                if current_weight in value_node.auxiliary['forward'] and value_node.op_type == "aten::_convolution":
                    behind = value_node.auxiliary['behind']
                    forwards = value_node.auxiliary['forward']
                    if isinstance(behind, str):
                        behind_all_layers += [behind]
                        forwards__all_layers += [forwards]
                    elif isinstance(behind, list):
                        behind_all_layers += behind
                        forwards__all_layers += forwards

                    else:
                        print("Useless behind", behind)

            behind_2_fws=[]   #['causal.conv1.weight', 'causal.conv1.bias', 'causal.conv1.weight', 'causal.conv1.bias']
            behind_2_fws_dict={}    #{causal.conv2.weight:'causal.conv1.weight', 'causal.conv1.bias'}
            for lay in forwards__all_layers:
                for key, value_node in self.atten_ops.items():
                    if lay in value_node.auxiliary['forward']:
                        behind_tmp = value_node.auxiliary['behind']
                        forwards_tmp = value_node.auxiliary['forward']
                        behind_2_fws_dict[lay]=forwards_tmp
                        if isinstance(forwards_tmp, str):
                            behind_2_fws += [forwards_tmp]
                        elif isinstance(forwards_tmp, list):
                            behind_2_fws += forwards_tmp

            com_layers=[]
            dif_layers=[]
            for e_fw in behind_2_fws:
                com_flag=True
                for k,v in behind_2_fws_dict.items():
                    if e_fw in v:
                        com_flag*=True
                    else:
                        com_flag*=False
                if com_flag:
                    com_layers.append(e_fw)
                else:
                    dif_layers.append(e_fw)

            return {"next": set(behind_all_layers), "current": set(com_layers)}


        else:
            """
             the others layer
            """
            print('*****Others layer******', current_weight)

            return self.warp_call(current_weight)

    def warp_call(self,current_weight):
        behind_layers = []
        forwards_layers = []
        fc_relaton_conv=[]
        assert isinstance(current_weight, str), "current_weight name must be string"

        for key, value_node in self.atten_ops.items():
            if current_weight in value_node.auxiliary['forward'] and value_node.op_type == "aten::_convolution":#  and value_node.op_type == "aten::_convolution"
                behind = value_node.auxiliary['behind']
                forwards = value_node.auxiliary['forward']
                if isinstance(behind, str):
                    behind_layers += [behind]
                    forwards_layers += [forwards]
                elif isinstance(behind, list):
                    behind_layers += behind
                    forwards_layers += forwards

                else:
                    print("Useless behind", behind)

            elif current_weight in value_node.auxiliary['forward'] and value_node.op_type == "aten::flatten":#  and value_node.op_type == "aten::_convolution"
                print("aten::flattbehind",value_node.auxiliary['behind'])
                print("aten::flatt_forward",value_node.auxiliary['forward'])
                fc_forward=value_node.auxiliary['forward']
                if isinstance(fc_forward, str):
                    fc_relaton_conv += [fc_forward]
                elif isinstance(fc_forward, list):
                    fc_relaton_conv += fc_forward


        #如果当前层下层链接到全连接层，则需要清理出，不进行压缩
        fc_flag=False
        for e_lay in fc_relaton_conv:
            if e_lay in forwards_layers:
                fc_flag=True

        if fc_flag:
            return {"next": {}, "current": {}}

        else:
            return {"next":set(behind_layers), "current":set(forwards_layers)}



class WarpVizGraph():
    """
    todo:to make better graph, this function can be down in some time.

    fun: wrap  torchviz.graph
    x = torch.randn(64,3,32,32)

    from torchviz import make_dot
    vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
    vis_graph.view()

    """

    pass



if __name__=="__main__":

    from models.resnet import WaveNet, ResNet18,vgg16
    from torchvision.models import resnet18
    # models = WaveNet(32, 32, 20, 128)
    # x = torch.randn(64, 20, 128)
    x = torch.randn(64, 3, 32,32)

    models=resnet18()
    trace = torch.jit.trace(models, x)


    AutoGraph_ = AutoGraph(trace)
    kone='layer4.0.conv2.weight'  #layer1.0.conv1.weight  conv1.weight  layer1.1.conv2.weight   causal.conv2.weight
    fwbws=AutoGraph_(kone)
    fws = fwbws["current"]
    bws = fwbws["next"]

    print(list(fws))
    print(bws)























