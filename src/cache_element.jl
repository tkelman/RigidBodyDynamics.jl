type CacheElement{T}
    dirty::Bool
    data::T
    CacheElement() = new(true)
    CacheElement(data::T) = new(true, data)
end

CacheElement{T}(data::T) = CacheElement{T}(data)

function update!{T}(element::CacheElement{T}, data::T)
    element.data = data
    element.dirty = false
end

function get(element::CacheElement)
    element.dirty && error("Cache dirty.")
    element.data
end

function setdirty!(element::CacheElement)
    element.dirty = true
end
