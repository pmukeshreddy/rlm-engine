interface MemoryDiffProps {
  before: Record<string, unknown> | null
  after: Record<string, unknown> | null
}

export default function MemoryDiff({ before, after }: MemoryDiffProps) {
  const beforeKeys = new Set(Object.keys(before || {}))
  const afterKeys = new Set(Object.keys(after || {}))
  const allKeys = new Set([...beforeKeys, ...afterKeys])

  if (allKeys.size === 0) {
    return (
      <div className="text-gray-500 text-sm">No memory changes</div>
    )
  }

  return (
    <div className="space-y-2">
      {Array.from(allKeys).map((key) => {
        const beforeValue = before?.[key]
        const afterValue = after?.[key]
        const isNew = !beforeKeys.has(key)
        const isDeleted = !afterKeys.has(key)
        const isModified = beforeKeys.has(key) && afterKeys.has(key) &&
          JSON.stringify(beforeValue) !== JSON.stringify(afterValue)

        return (
          <div
            key={key}
            className={`p-2 rounded text-sm font-mono ${
              isNew
                ? 'bg-green-50 border border-green-200'
                : isDeleted
                ? 'bg-red-50 border border-red-200'
                : isModified
                ? 'bg-yellow-50 border border-yellow-200'
                : 'bg-gray-50 border border-gray-200'
            }`}
          >
            <div className="font-semibold text-gray-700">{key}</div>
            {isNew && (
              <div className="text-green-700">
                + {JSON.stringify(afterValue, null, 2)}
              </div>
            )}
            {isDeleted && (
              <div className="text-red-700">
                - {JSON.stringify(beforeValue, null, 2)}
              </div>
            )}
            {isModified && (
              <>
                <div className="text-red-700">
                  - {JSON.stringify(beforeValue, null, 2)}
                </div>
                <div className="text-green-700">
                  + {JSON.stringify(afterValue, null, 2)}
                </div>
              </>
            )}
            {!isNew && !isDeleted && !isModified && (
              <div className="text-gray-600">
                {JSON.stringify(afterValue, null, 2)}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
