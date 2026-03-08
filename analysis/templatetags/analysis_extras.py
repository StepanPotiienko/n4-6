from django import template

register = template.Library()


@register.filter
def get_item(mapping, key):
    if mapping is None:
        return None
    try:
        return mapping.get(key)
    except AttributeError:
        return None


@register.filter
def is_equal(value, arg):
    """Checks if two values are equal after string conversion."""
    return str(value) == str(arg)
