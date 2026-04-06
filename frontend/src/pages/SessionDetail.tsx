import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getSession, getSessionMemory, getExecutions } from '../api/client'
import { formatDistanceToNow } from 'date-fns'
import api from '../api/client'

export default function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const queryClient = useQueryClient()
  const [newKey, setNewKey] = useState('')
  const [newValue, setNewValue] = useState('')
  const [showAddMemory, setShowAddMemory] = useState(false)

  const { data: session, isLoading: sessionLoading } = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => getSession(sessionId!),
    enabled: !!sessionId,
  })

  const { data: memory } = useQuery({
    queryKey: ['sessionMemory', sessionId],
    queryFn: () => getSessionMemory(sessionId!),
    enabled: !!sessionId,
  })

  const { data: executions } = useQuery({
    queryKey: ['executions', sessionId],
    queryFn: () => getExecutions(sessionId),
    enabled: !!sessionId,
  })

  const addMemoryMutation = useMutation({
    mutationFn: async ({ key, value }: { key: string; value: string }) => {
      let parsedValue: unknown = value
      try {
        parsedValue = JSON.parse(value)
      } catch {
        // Keep as string if not valid JSON
      }
      await api.post(`/sessions/${sessionId}/memory`, { key, value: parsedValue })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessionMemory', sessionId] })
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] })
      setNewKey('')
      setNewValue('')
      setShowAddMemory(false)
    },
  })

  const deleteMemoryMutation = useMutation({
    mutationFn: async (key: string) => {
      await api.delete(`/sessions/${sessionId}/memory/${key}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessionMemory', sessionId] })
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] })
    },
  })

  if (sessionLoading) {
    return <div className="text-center py-8">Loading...</div>
  }

  if (!session) {
    return <div className="text-center py-8">Session not found</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">
            {session.name || `Session ${session.id.slice(0, 8)}`}
          </h1>
          <p className="text-gray-500 mt-1">
            Created {formatDistanceToNow(new Date(session.created_at), { addSuffix: true })}
          </p>
        </div>
        <Link
          to={`/new?session=${sessionId}`}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          New Execution
        </Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Context Info */}
        <div className="bg-white rounded-lg border p-6">
          <h2 className="font-semibold mb-4">Context</h2>
          {session.context_metadata?.size ? (
            <div className="space-y-2">
              <p>
                <span className="text-gray-500">Size:</span>{' '}
                {(session.context_metadata.size as number).toLocaleString()} characters
              </p>
              <p>
                <span className="text-gray-500">Hash:</span>{' '}
                <code className="text-xs bg-gray-100 px-1 rounded">
                  {(session.context_metadata.hash as string)?.slice(0, 16)}...
                </code>
              </p>
            </div>
          ) : (
            <p className="text-gray-500">No context stored</p>
          )}
        </div>

        {/* Memory */}
        <div className="bg-white rounded-lg border p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="font-semibold">Memory ({Object.keys(memory || {}).length} keys)</h2>
            <button
              onClick={() => setShowAddMemory(!showAddMemory)}
              className="text-sm px-3 py-1 bg-primary-600 text-white rounded hover:bg-primary-700"
            >
              + Add
            </button>
          </div>
          
          {showAddMemory && (
            <div className="mb-4 p-3 bg-gray-50 rounded space-y-2">
              <input
                type="text"
                placeholder="Key"
                value={newKey}
                onChange={(e) => setNewKey(e.target.value)}
                className="w-full px-2 py-1 border rounded text-sm"
              />
              <input
                type="text"
                placeholder="Value (string or JSON)"
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                className="w-full px-2 py-1 border rounded text-sm"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => addMemoryMutation.mutate({ key: newKey, value: newValue })}
                  disabled={!newKey || addMemoryMutation.isPending}
                  className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50"
                >
                  Save
                </button>
                <button
                  onClick={() => setShowAddMemory(false)}
                  className="px-3 py-1 border rounded text-sm hover:bg-gray-100"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
          
          {memory && Object.keys(memory).length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-auto">
              {Object.entries(memory).map(([key, value]) => (
                <div key={key} className="p-2 bg-gray-50 rounded flex justify-between items-start group">
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm">{key}</p>
                    <p className="text-xs text-gray-600 font-mono truncate">
                      {JSON.stringify(value)}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      if (confirm(`Delete memory key "${key}"?`)) {
                        deleteMemoryMutation.mutate(key)
                      }
                    }}
                    className="ml-2 text-red-500 opacity-0 group-hover:opacity-100 text-xs"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500">No memory stored</p>
          )}
        </div>
      </div>

      {/* Executions */}
      <div className="bg-white rounded-lg border">
        <div className="p-4 border-b">
          <h2 className="font-semibold">Executions ({executions?.total || 0})</h2>
        </div>
        <div className="divide-y">
          {!executions?.executions.length ? (
            <div className="p-8 text-center text-gray-500">
              No executions yet
            </div>
          ) : (
            executions.executions.map((execution) => (
              <Link
                key={execution.id}
                to={`/executions/${execution.id}`}
                className="p-4 flex items-center gap-4 hover:bg-gray-50"
              >
                <div
                  className={`w-2 h-2 rounded-full ${
                    execution.status === 'completed'
                      ? 'bg-green-500'
                      : execution.status === 'failed'
                      ? 'bg-red-500'
                      : 'bg-gray-400'
                  }`}
                />
                <div className="flex-1">
                  <p className="font-medium truncate">{execution.user_query}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm">${execution.total_cost_usd.toFixed(4)}</p>
                  <p className="text-xs text-gray-500">
                    {execution.started_at
                      ? formatDistanceToNow(new Date(execution.started_at), {
                          addSuffix: true,
                        })
                      : 'Pending'}
                  </p>
                </div>
              </Link>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
