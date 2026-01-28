import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getSession, getSessionMemory, getExecutions } from '../api/client'
import { formatDistanceToNow } from 'date-fns'

export default function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>()

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
          <h2 className="font-semibold mb-4">Memory ({session.memory_count} keys)</h2>
          {memory && Object.keys(memory).length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-auto">
              {Object.entries(memory).map(([key, value]) => (
                <div key={key} className="p-2 bg-gray-50 rounded">
                  <p className="font-medium text-sm">{key}</p>
                  <p className="text-xs text-gray-600 font-mono truncate">
                    {JSON.stringify(value)}
                  </p>
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
