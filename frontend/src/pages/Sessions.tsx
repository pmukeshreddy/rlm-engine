import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { PlusIcon, TrashIcon } from '@heroicons/react/24/outline'
import { getSessions, createSession, deleteSession } from '../api/client'
import { formatDistanceToNow } from 'date-fns'

export default function Sessions() {
  const queryClient = useQueryClient()
  const [showNewForm, setShowNewForm] = useState(false)
  const [newName, setNewName] = useState('')
  const [newContext, setNewContext] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: getSessions,
  })

  const createMutation = useMutation({
    mutationFn: createSession,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
      setShowNewForm(false)
      setNewName('')
      setNewContext('')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
    },
  })

  const handleCreate = () => {
    createMutation.mutate({
      name: newName || undefined,
      context: newContext || undefined,
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Sessions</h1>
          <p className="text-gray-500 mt-1">
            Manage your sessions and stored contexts
          </p>
        </div>
        <button
          onClick={() => setShowNewForm(!showNewForm)}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          <PlusIcon className="w-5 h-5" />
          New Session
        </button>
      </div>

      {/* New Session Form */}
      {showNewForm && (
        <div className="bg-white rounded-lg border p-6 space-y-4">
          <h3 className="font-semibold">Create New Session</h3>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Name (optional)
            </label>
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              placeholder="My Analysis Session"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Context (optional - can add later)
            </label>
            <textarea
              value={newContext}
              onChange={(e) => setNewContext(e.target.value)}
              rows={6}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
              placeholder="Paste your large context here..."
            />
            {newContext && (
              <p className="text-sm text-gray-500 mt-1">
                {newContext.length.toLocaleString()} characters
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              disabled={createMutation.isPending}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
            >
              {createMutation.isPending ? 'Creating...' : 'Create Session'}
            </button>
            <button
              onClick={() => setShowNewForm(false)}
              className="px-4 py-2 border rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="bg-white rounded-lg border">
        {isLoading ? (
          <div className="p-8 text-center text-gray-500">Loading...</div>
        ) : !data?.sessions.length ? (
          <div className="p-8 text-center text-gray-500">
            No sessions yet. Create one to get started.
          </div>
        ) : (
          <div className="divide-y">
            {data.sessions.map((session) => (
              <div
                key={session.id}
                className="p-4 flex items-center gap-4 hover:bg-gray-50"
              >
                <Link to={`/sessions/${session.id}`} className="flex-1">
                  <p className="font-medium">
                    {session.name || `Session ${session.id.slice(0, 8)}`}
                  </p>
                  <div className="flex gap-4 text-sm text-gray-500 mt-1">
                    <span>
                      {session.context_metadata?.size
                        ? `${(session.context_metadata.size as number).toLocaleString()} chars`
                        : 'No context'}
                    </span>
                    <span>{session.memory_count} memories</span>
                    <span>
                      Updated{' '}
                      {formatDistanceToNow(new Date(session.updated_at), {
                        addSuffix: true,
                      })}
                    </span>
                  </div>
                </Link>
                <button
                  onClick={() => {
                    if (confirm('Delete this session?')) {
                      deleteMutation.mutate(session.id)
                    }
                  }}
                  className="p-2 text-gray-400 hover:text-red-500"
                >
                  <TrashIcon className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
