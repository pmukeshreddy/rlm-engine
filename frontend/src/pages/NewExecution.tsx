import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { getSessions, getSession, createExecution } from '../api/client'

export default function NewExecution() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const preselectedSession = searchParams.get('session')

  const [query, setQuery] = useState('')
  const [context, setContext] = useState('')
  const [sessionId, setSessionId] = useState(preselectedSession || '')
  const [model, setModel] = useState('')
  const [useSessionContext, setUseSessionContext] = useState(!!preselectedSession)

  const { data: sessions } = useQuery({
    queryKey: ['sessions'],
    queryFn: getSessions,
  })

  const { data: selectedSession } = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => getSession(sessionId),
    enabled: !!sessionId && useSessionContext,
  })

  const executeMutation = useMutation({
    mutationFn: createExecution,
    onSuccess: (data) => {
      navigate(`/executions/${data.id}`)
    },
  })

  useEffect(() => {
    if (preselectedSession) {
      setSessionId(preselectedSession)
      setUseSessionContext(true)
    }
  }, [preselectedSession])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    executeMutation.mutate({
      user_query: query,
      context: useSessionContext ? undefined : context,
      session_id: useSessionContext && sessionId ? sessionId : undefined,
      model: model || undefined,
    })
  }

  const contextSize = useSessionContext
    ? (selectedSession?.context_metadata?.size as number) || 0
    : context.length

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold">New Execution</h1>
        <p className="text-gray-500 mt-1">
          Process a large context with the RLM Engine
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Query */}
        <div className="bg-white rounded-lg border p-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Query / Task
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            required
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            placeholder="What would you like to do with this context? e.g., 'Summarize the key findings' or 'Extract all mentioned dates and events'"
          />
        </div>

        {/* Context Source */}
        <div className="bg-white rounded-lg border p-6 space-y-4">
          <h3 className="font-medium">Context Source</h3>
          
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={!useSessionContext}
                onChange={() => setUseSessionContext(false)}
                className="text-primary-600"
              />
              <span>Paste context directly</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={useSessionContext}
                onChange={() => setUseSessionContext(true)}
                className="text-primary-600"
              />
              <span>Use session context</span>
            </label>
          </div>

          {useSessionContext ? (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Select Session
              </label>
              <select
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="">Select a session...</option>
                {sessions?.sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name || `Session ${s.id.slice(0, 8)}`} (
                    {s.context_metadata?.size
                      ? `${(s.context_metadata.size as number).toLocaleString()} chars`
                      : 'no context'}
                    )
                  </option>
                ))}
              </select>
              {selectedSession && !selectedSession.context_metadata?.size && (
                <p className="text-red-500 text-sm mt-1">
                  This session has no stored context. Please select another or paste context directly.
                </p>
              )}
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Context
              </label>
              <textarea
                value={context}
                onChange={(e) => setContext(e.target.value)}
                rows={10}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
                placeholder="Paste your large context here (500K+ characters supported)..."
              />
            </div>
          )}

          {contextSize > 0 && (
            <p className="text-sm text-gray-500">
              Context size: {contextSize.toLocaleString()} characters
              {contextSize > 100000 && (
                <span className="text-green-600 ml-2">
                  (Will be processed in chunks)
                </span>
              )}
            </p>
          )}
        </div>

        {/* Advanced Options */}
        <div className="bg-white rounded-lg border p-6">
          <h3 className="font-medium mb-4">Advanced Options</h3>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model (optional)
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">Default (gpt-4-turbo-preview)</option>
              <option value="gpt-4o">GPT-4o</option>
              <option value="gpt-4o-mini">GPT-4o Mini (cheaper)</option>
              <option value="gpt-4-turbo-preview">GPT-4 Turbo</option>
              <option value="claude-3-opus-20240229">Claude 3 Opus</option>
              <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
              <option value="claude-3-haiku-20240307">Claude 3 Haiku (cheaper)</option>
            </select>
          </div>
        </div>

        {/* Submit */}
        <div className="flex gap-4">
          <button
            type="submit"
            disabled={
              executeMutation.isPending ||
              !query ||
              (!useSessionContext && !context) ||
              (useSessionContext && (!sessionId || !selectedSession?.context_metadata?.size))
            }
            className="flex-1 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {executeMutation.isPending ? 'Processing...' : 'Run Execution'}
          </button>
        </div>

        {executeMutation.isError && (
          <div className="bg-red-50 text-red-700 p-4 rounded-lg">
            Error: {(executeMutation.error as Error).message}
          </div>
        )}
      </form>
    </div>
  )
}
