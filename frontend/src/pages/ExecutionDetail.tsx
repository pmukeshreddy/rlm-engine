import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline'
import { getExecution } from '../api/client'
import type { ExecutionNode } from '../api/client'
import ExecutionTree from '../components/ExecutionTree'
import CodeBlock from '../components/CodeBlock'
import MemoryDiff from '../components/MemoryDiff'
import { format } from 'date-fns'

export default function ExecutionDetail() {
  const { executionId } = useParams<{ executionId: string }>()
  const [selectedNode, setSelectedNode] = useState<ExecutionNode | null>(null)

  const { data: execution, isLoading } = useQuery({
    queryKey: ['execution', executionId],
    queryFn: () => getExecution(executionId!),
    enabled: !!executionId,
  })

  if (isLoading) {
    return <div className="text-center py-8">Loading...</div>
  }

  if (!execution) {
    return <div className="text-center py-8">Execution not found</div>
  }

  const statusIcons: Record<string, JSX.Element> = {
    completed: <CheckCircleIcon className="w-6 h-6 text-green-500" />,
    failed: <XCircleIcon className="w-6 h-6 text-red-500" />,
    running: <ClockIcon className="w-6 h-6 text-yellow-500 animate-spin" />,
    pending: <ClockIcon className="w-6 h-6 text-gray-400" />,
    cancelled: <XCircleIcon className="w-6 h-6 text-gray-400" />,
  }
  const statusIcon = statusIcons[execution.status] || <ClockIcon className="w-6 h-6 text-gray-400" />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-3">
            {statusIcon}
            <h1 className="text-2xl font-bold">Execution Details</h1>
          </div>
          <p className="text-gray-500 mt-1 max-w-2xl">{execution.user_query}</p>
        </div>
        {execution.session_id && (
          <Link
            to={`/sessions/${execution.session_id}`}
            className="text-primary-600 hover:underline text-sm"
          >
            View Session
          </Link>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">Context Size</p>
          <p className="text-xl font-bold">
            {execution.context_size.toLocaleString()}
          </p>
          <p className="text-xs text-gray-400">characters</p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">Total Tokens</p>
          <p className="text-xl font-bold">
            {(execution.total_input_tokens + execution.total_output_tokens).toLocaleString()}
          </p>
          <p className="text-xs text-gray-400">
            {execution.total_input_tokens.toLocaleString()} in / {execution.total_output_tokens.toLocaleString()} out
          </p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">Total Cost</p>
          <p className="text-xl font-bold">${execution.total_cost_usd.toFixed(4)}</p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">Duration</p>
          <p className="text-xl font-bold">
            {execution.started_at && execution.completed_at
              ? `${((new Date(execution.completed_at).getTime() - new Date(execution.started_at).getTime()) / 1000).toFixed(1)}s`
              : 'N/A'}
          </p>
          {execution.started_at && (
            <p className="text-xs text-gray-400">
              {format(new Date(execution.started_at), 'MMM d, HH:mm:ss')}
            </p>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Tree and Code */}
        <div className="space-y-6">
          {/* Execution Tree */}
          <ExecutionTree
            tree={execution.tree?.tree || null}
            selectedNode={selectedNode}
            onSelectNode={setSelectedNode}
          />

          {/* Generated Code */}
          <div className="bg-white rounded-lg border p-4">
            <h3 className="font-semibold mb-4">Generated Code</h3>
            {execution.generated_code ? (
              <CodeBlock code={execution.generated_code} />
            ) : (
              <p className="text-gray-500">No code generated</p>
            )}
          </div>
        </div>

        {/* Right: Result and Node Details */}
        <div className="space-y-6">
          {/* Final Result or Error */}
          <div className="bg-white rounded-lg border p-4">
            <h3 className="font-semibold mb-4">
              {execution.status === 'completed' ? 'Final Result' : 'Error'}
            </h3>
            {execution.final_result ? (
              <div className="prose prose-sm max-w-none">
                <pre className="whitespace-pre-wrap bg-gray-50 p-4 rounded text-sm">
                  {execution.final_result}
                </pre>
              </div>
            ) : execution.error_message ? (
              <div className="bg-red-50 text-red-700 p-4 rounded text-sm font-mono whitespace-pre-wrap">
                {execution.error_message}
              </div>
            ) : (
              <p className="text-gray-500">No result</p>
            )}
          </div>

          {/* Selected Node Details */}
          {selectedNode && (
            <div className="bg-white rounded-lg border p-4">
              <h3 className="font-semibold mb-4">
                {selectedNode.node_type === 'root' ? 'Root Agent' : `Child Agent #${selectedNode.sequence_number + 1}`}
              </h3>
              
              <div className="space-y-4">
                {/* Prompt */}
                {selectedNode.prompt && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-1">Prompt</h4>
                    <pre className="bg-gray-50 p-2 rounded text-sm whitespace-pre-wrap max-h-40 overflow-auto">
                      {selectedNode.prompt}
                    </pre>
                  </div>
                )}

                {/* Output */}
                {selectedNode.output && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-1">Output</h4>
                    <pre className="bg-gray-50 p-2 rounded text-sm whitespace-pre-wrap max-h-40 overflow-auto">
                      {selectedNode.output}
                    </pre>
                  </div>
                )}

                {/* Stats */}
                <div className="flex gap-4 text-sm">
                  <span className="text-gray-500">
                    Tokens: {selectedNode.input_tokens + selectedNode.output_tokens}
                  </span>
                  <span className="text-gray-500">
                    Cost: ${selectedNode.cost_usd.toFixed(4)}
                  </span>
                  {selectedNode.model_used && (
                    <span className="text-gray-500">
                      Model: {selectedNode.model_used}
                    </span>
                  )}
                </div>

                {/* Memory Diff */}
                {(selectedNode.memory_before || selectedNode.memory_after) && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-1">Memory Changes</h4>
                    <MemoryDiff
                      before={selectedNode.memory_before}
                      after={selectedNode.memory_after}
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
