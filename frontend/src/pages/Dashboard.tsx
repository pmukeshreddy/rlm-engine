import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  CurrencyDollarIcon,
  DocumentTextIcon,
  ClockIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline'
import { getStats, getExecutions } from '../api/client'
import StatsCard from '../components/StatsCard'
import { formatDistanceToNow } from 'date-fns'

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: () => getStats(),
  })

  const { data: executions, isLoading: executionsLoading } = useQuery({
    queryKey: ['executions'],
    queryFn: () => getExecutions(),
  })

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Overview of your RLM Engine usage
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Executions"
          value={statsLoading ? '...' : stats?.total_executions || 0}
          icon={<DocumentTextIcon className="w-6 h-6" />}
        />
        <StatsCard
          title="Total Cost"
          value={statsLoading ? '...' : `$${(stats?.total_cost_usd || 0).toFixed(2)}`}
          subtitle={`Avg: $${(stats?.average_cost_per_execution || 0).toFixed(4)}/exec`}
          icon={<CurrencyDollarIcon className="w-6 h-6" />}
        />
        <StatsCard
          title="Total Tokens"
          value={
            statsLoading
              ? '...'
              : ((stats?.total_input_tokens || 0) + (stats?.total_output_tokens || 0)).toLocaleString()
          }
          subtitle={`${(stats?.total_input_tokens || 0).toLocaleString()} in / ${(stats?.total_output_tokens || 0).toLocaleString()} out`}
          icon={<ClockIcon className="w-6 h-6" />}
        />
        <StatsCard
          title="Success Rate"
          value={
            statsLoading
              ? '...'
              : stats?.total_executions
              ? `${(((stats.executions_by_status?.completed || 0) / stats.total_executions) * 100).toFixed(0)}%`
              : 'N/A'
          }
          icon={<CheckCircleIcon className="w-6 h-6" />}
        />
      </div>

      {/* Recent Executions */}
      <div className="bg-white rounded-lg border">
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="font-semibold">Recent Executions</h2>
          <Link
            to="/new"
            className="text-sm text-primary-600 hover:text-primary-700"
          >
            New Execution
          </Link>
        </div>
        <div className="divide-y">
          {executionsLoading ? (
            <div className="p-8 text-center text-gray-500">Loading...</div>
          ) : !executions?.executions.length ? (
            <div className="p-8 text-center text-gray-500">
              No executions yet.{' '}
              <Link to="/new" className="text-primary-600 hover:underline">
                Create one
              </Link>
            </div>
          ) : (
            executions.executions.slice(0, 10).map((execution) => (
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
                      : execution.status === 'running'
                      ? 'bg-yellow-500'
                      : 'bg-gray-400'
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{execution.user_query}</p>
                  <p className="text-sm text-gray-500">
                    {execution.context_size.toLocaleString()} chars
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">
                    ${execution.total_cost_usd.toFixed(4)}
                  </p>
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
