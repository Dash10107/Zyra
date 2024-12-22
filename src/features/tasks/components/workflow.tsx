import React, { useCallback, useState, useRef } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  Background,
  Controls,
  MiniMap,
  Handle,
  Node,
  Edge,
  Connection,
  NodeChange,
  EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Define the type for the Custom Node's props
type CustomNodeProps = {
  data: {
    label: string;
  };
};

const CustomNode: React.FC<CustomNodeProps> = ({ data }) => {
  return (
    <div
      style={{
        padding: 20,
        border: '2px solid #4CAF50',
        borderRadius: 10,
        background: 'linear-gradient(135deg, #6e7dff, #3a86ff)',
        color: 'white',
        fontWeight: 'bold',
        fontSize: '14px',
        width: '200px',
        textAlign: 'center',
      }}
    >
      <div>{data.label}</div>
      <Handle type="source" position="right" style={{ background: '#fff' }} />
      <Handle type="target" position="left" style={{ background: '#fff' }} />
    </div>
  );
};

const Workflow: React.FC = () => {
  const initialNodes: Node[] = [
    {
      id: '1',
      type: 'customNode',
      position: { x: 250, y: 5 },
      data: { label: 'Prepare Monthly Report' },
    },
    {
      id: '2',
      type: 'customNode',
      position: { x: 100, y: 200 },
      data: { label: 'Create Marketing Campaign Plan' },
    },
    {
      id: '3',
      type: 'customNode',
      position: { x: 400, y: 200 },
      data: { label: 'Design Landing Page' },
    },
    {
      id: '4',
      type: 'customNode',
      position: { x: 250, y: 400 },
      data: { label: "Review Team's Code Submissions" },
    },
  ];

  const initialEdges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2', type: 'smoothstep' },
    { id: 'e2-3', source: '2', target: '3', type: 'smoothstep' },
    { id: 'e3-4', source: '3', target: '4', type: 'smoothstep' },
  ];

  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);
  const [newNodeLabel, setNewNodeLabel] = useState<string>(''); // Text input for new node label
  const [isAddingNode, setIsAddingNode] = useState<boolean>(false); // Whether the user is in "add node" mode

  const newNodePosition = useRef<{ x: number; y: number }>({ x: 100, y: 100 });

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge(connection, eds));
    },
    []
  );

  // Function to add a new node with the text input label
  const addNewNode = () => {
    if (!newNodeLabel.trim()) return; // Prevent adding a node if label is empty

    const newNode: Node = {
      id: `${nodes.length + 1}`,
      type: 'customNode',
      position: { ...newNodePosition.current },
      data: { label: newNodeLabel },
    };

    setNodes((prevNodes) => [...prevNodes, newNode]);
    setNewNodeLabel(''); // Clear input field after adding node
    setIsAddingNode(false); // Reset to normal mode
  };

  // Function to handle the position update of the new node when dragging
  const onNodeDrag = (event: any, node: Node) => {
    if (isAddingNode) {
      newNodePosition.current = { x: node.position.x, y: node.position.y };
    }
  };

  const nodeTypes = { customNode: CustomNode };

  return (
    <div style={{ height: '100vh', background: '#fff', position: 'relative' }}>
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          connectionLineStyle={{ stroke: 'rgb(255, 255, 255)', strokeWidth: 2 }}
          onNodeDrag={onNodeDrag} // Track dragging of nodes
        >
          <Controls />
          <MiniMap />
          <Background color="#aaa" gap={20} />
        </ReactFlow>

        {/* Add Node Button */}
        <button
          onClick={() => setIsAddingNode(true)}
          style={{
            position: 'absolute',
            top: 20,
            left: 20,
            padding: '10px 20px',
            backgroundColor: '#4CAF50',
            color: 'white',
            borderRadius: '5px',
            border: 'none',
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          Add New Node
        </button>

        {/* Node Label Input */}
        {isAddingNode && (
          <div
            style={{
              position: 'absolute',
              top: 20,
              left: 150,
              background: '#fff',
              padding: '10px 20px',
              borderRadius: '5px',
              border: '2px solid #4CAF50',
            }}
          >
            <input
              type="text"
              placeholder="Enter node label"
              value={newNodeLabel}
              onChange={(e) => setNewNodeLabel(e.target.value)}
              style={{
                padding: '10px',
                width: '200px',
                marginBottom: '10px',
              }}
            />
            <button
              onClick={addNewNode}
              style={{
                padding: '5px 15px',
                backgroundColor: '#4CAF50',
                color: 'white',
                borderRadius: '5px',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              Add Node
            </button>
          </div>
        )}
      </ReactFlowProvider>
    </div>
  );
};

export default Workflow;