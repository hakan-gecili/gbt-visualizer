const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const text = await response.text()
    let message = text
    try {
      const parsed = JSON.parse(text) as { detail?: string }
      message = parsed.detail ?? text
    } catch {
      // keep raw response text
    }
    throw new Error(message || `Request failed with status ${response.status}`)
  }
  return (await response.json()) as T
}

export function createFormData(values: Record<string, string | File>): FormData {
  const formData = new FormData()
  Object.entries(values).forEach(([key, value]) => {
    formData.append(key, value)
  })
  return formData
}
