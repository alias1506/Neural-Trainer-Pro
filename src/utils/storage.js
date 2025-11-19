export function saveSession(entry) {
  const key = 'cifar10_sessions'
  const list = JSON.parse(localStorage.getItem(key) || '[]')
  list.unshift(entry)
  localStorage.setItem(key, JSON.stringify(list.slice(0,50)))
}
export function listSessions() {
  return JSON.parse(localStorage.getItem('cifar10_sessions') || '[]')
}
export function deleteSession(id) {
  const key = 'cifar10_sessions'
  const list = JSON.parse(localStorage.getItem(key) || '[]')
  const filtered = list.filter(session => session.id !== id)
  localStorage.setItem(key, JSON.stringify(filtered))
  return filtered
}