import { useState } from 'react'
import {
  ChevronDownIcon,
  ChevronRightIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline'
import type { ExecutionNode } from '../api/client'

interface TreeNodeProps {
  node: ExecutionNode
  isSelected: boolean
  onSelect: (node: ExecutionNode) => void
}

function TreeNode({ node, isSelected, onSelect }: TreeNodeProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const hasChildren = node.children && node.children.length > 0

  const statusIcon = {
    completed: <CheckCircleIcon className="w-4 h-4 text-green-500" />,
    failed: <XCircleIcon className="w-4 h-4 text-red-500" />,
    running: <ClockIcon className="w-4 h-4 text-yellow-500 animate-spin" />,
    pending: <ClockIcon className="w-4 h-4 text-gray-400" />,
  }[node.status] || <ClockIcon className="w-4 h-4 text-gray-400" />

  return (
    <div className="select-none">
      <div
        className={`flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-gray-100 ${
          isSelected ? 'bg-primary-50 border border-primary-200' : ''
        }`}
        onClick={() => onSelect(node)}
      >
        {hasChildren ? (
          <button
            onClick={(e) => {
              e.stopPropagation()
              setIsExpanded(!isExpanded)
            }}
            className="p-0.5 hover:bg-gray-200 rounded"
          >
            {isExpanded ? (
              <ChevronDownIcon className="w-4 h-4" />
            ) : (
              <ChevronRightIcon className="w-4 h-4" />
            )}
          </button>
        ) : (
          <span className="w-5" />
        )}
        {statusIcon}
        <span className="font-medium text-sm">
          {node.node_type === 'root' ? 'Root Agent' : `Child #${node.sequence_number + 1}`}
        </span>
        <span className="text-xs text-gray-500">
          {node.input_tokens + node.output_tokens} tokens
        </span>
        <span className="text-xs text-gray-500">
          ${node.cost_usd.toFixed(4)}
        </span>
      </div>
      {hasChildren && isExpanded && (
        <div className="ml-6 border-l border-gray-200">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              isSelected={isSelected}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

interface ExecutionTreeProps {
  tree: ExecutionNode | null
  selectedNode: ExecutionNode | null
  onSelectNode: (node: ExecutionNode) => void
}

export default function ExecutionTree({
  tree,
  selectedNode,
  onSelectNode,
}: ExecutionTreeProps) {
  if (!tree) {
    return (
      <div className="text-center text-gray-500 py-8">
        No execution tree available
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg border p-4">
      <h3 className="font-semibold mb-4">Execution Tree</h3>
      <TreeNode
        node={tree}
        isSelected={selectedNode?.id === tree.id}
        onSelect={onSelectNode}
      />
    </div>
  )
}
